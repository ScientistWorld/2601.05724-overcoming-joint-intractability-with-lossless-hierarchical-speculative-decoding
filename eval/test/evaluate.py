"""Held-out (test) slice evaluator for speculative-decoding verifier programs.

Self-contained: the hidden target/draft instances are generated here and
nothing is imported from data/, method/ or eval/train/.

A *method* is a directory ``checkpoints/<name>/`` containing ``verifier.py``.
The evaluator runs every such program in its own subprocess and streams it
verification problems one at a time (JSON lines; protocol in
``briefing/evaluation.md``).  A problem contains only what a verifier can
access in speculative decoding: the K draft blocks, the target rows
p(.|context, X_{1:t}) for t = 0..gamma and the draft rows q(.|context, X_{1:t})
for t = 0..gamma-1 along every draft.  The program answers with the
probability of each outcome (draft index k, accepted length tau, next token
y), meaning that one target step emits ``drafts[k][:tau] + [y]``.

Scoring is exact.  For every context the evaluator enumerates every possible
draft tuple (weighted by its draft probability), so

* ``block_efficiency`` = expected number of emitted tokens per target step,
  averaged over the experiment's contexts (no Monte-Carlo noise);
* losslessness is checked exactly: the emitted prefix followed by target
  sampling must reproduce p over gamma + 1 tokens (completion total
  variation <= LOSSLESS_TOL for every context).

Any protocol violation, non-finite/negative/unnormalised probability,
non-deterministic answer, crash, timeout or lossy context marks the method
as failed; a failed method scores ``block_efficiency = 0.0`` in every
experiment (below the 1.0 that any valid decoder attains).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Hidden test instances: same model families and settings as the visible
# slice (data/synthetic_lm.py), different salt and seed, more contexts.
TEST_SALT = "hsd-gym-heldout-test-v2-7c1e9a"
TEST_SEED = 91138417
TEST_CONTEXTS_PER_CONFIG: dict[str, int] = {
    "single_draft_main": 4,
    "single_draft_temperature": 4,
    "single_draft_length": 4,
    "multi_draft": 2,
}

REGIMES: dict[str, float] = {
    "peaked": 3.0,
    "medium": 2.0,
    "flat": 1.2,
}

# Draft/target pairings (analogues of pairing one small draft with targets of
# different sizes).
DRAFT_PAIRS: dict[str, tuple[float, float]] = {
    "aligned": (0.85, 0.6),
    "misaligned": (0.7, 1.1),
}


def _rng(*key) -> random.Random:
    digest = hashlib.sha256(repr(key).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _softmax(logits: list[float], temperature: float) -> list[float]:
    top = max(logits)
    exps = [math.exp((v - top) / temperature) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


def node_distributions(
    salt: str,
    seed: int,
    regime: str,
    pair: str,
    vocab: int,
    temperature: float,
    node: tuple[int, ...],
) -> tuple[list[float], list[float]]:
    """Return (p(.|node), q(.|node)) for one node of an instance."""
    rng = _rng(salt, seed, regime, pair, vocab, node)
    scale = REGIMES[regime]
    shrink, noise = DRAFT_PAIRS[pair]
    z = [rng.gauss(0.0, 1.0) * scale for _ in range(vocab)]
    e = [rng.gauss(0.0, 1.0) for _ in range(vocab)]
    p = _softmax(z, temperature)
    q = _softmax([shrink * a + noise * b for a, b in zip(z, e)], temperature)
    return p, q


def experiment_configs() -> dict[str, list[dict]]:
    """Benchmark settings shared by the train and test slices."""
    configs: dict[str, list[dict]] = {
        "single_draft_main": [],
        "single_draft_temperature": [],
        "single_draft_length": [],
        "multi_draft": [],
    }
    for regime in REGIMES:
        for pair in DRAFT_PAIRS:
            configs["single_draft_main"].append(
                dict(regime=regime, pair=pair, vocab=6, gamma=4, drafts=1, temperature=1.0)
            )
        for temperature in (0.6, 0.8):
            configs["single_draft_temperature"].append(
                dict(regime=regime, pair="aligned", vocab=6, gamma=4, drafts=1, temperature=temperature)
            )
        for gamma in (2, 3, 4, 5):
            configs["single_draft_length"].append(
                dict(regime=regime, pair="aligned", vocab=5, gamma=gamma, drafts=1, temperature=1.0)
            )
        for vocab, drafts, gamma in ((3, 2, 4), (4, 2, 3), (3, 3, 3)):
            for pair in DRAFT_PAIRS:
                configs["multi_draft"].append(
                    dict(regime=regime, pair=pair, vocab=vocab, gamma=gamma, drafts=drafts, temperature=1.0)
                )
    return configs


def sample_contexts(salt: str, seed: int, config: dict, count: int) -> list[tuple[int, ...]]:
    """Contexts reached by sampling the target model from an empty prompt."""
    contexts = []
    for index in range(count):
        rng = _rng(salt, seed, "context", sorted(config.items()), index)
        length = 2 + rng.randrange(5)
        ctx: tuple[int, ...] = (index,)  # distinct prompt token per context
        for _ in range(length):
            p, _q = node_distributions(
                salt, seed, config["regime"], config["pair"], config["vocab"],
                config["temperature"], ctx,
            )
            u = rng.random()
            acc = 0.0
            tok = len(p) - 1
            for i, v in enumerate(p):
                acc += v
                if u <= acc:
                    tok = i
                    break
            ctx = ctx + (tok,)
        contexts.append(ctx)
    return contexts


SLICE = "test"
# Order in which problems are streamed (exact scores do not depend on it).
SHUFFLE_SEED = int(os.environ.get("EVAL_STREAM_SEED", "4049"))
LOSSLESS_TOL = 1e-9
MASS_TOL = 1e-9
PROBE_FRACTION = 0.02
PROBE_TOL = 1e-12
TIME_BUDGET_S = float(os.environ.get("VERIFIER_TIME_BUDGET_S", "5400"))
FAILED_SCORE = 0.0


# --------------------------------------------------------------------------
# Instances
# --------------------------------------------------------------------------


def config_label(cfg: dict) -> str:
    return (
        f"{cfg['regime']}/{cfg['pair']}/V{cfg['vocab']}/g{cfg['gamma']}"
        f"/K{cfg['drafts']}/T{cfg['temperature']}"
    )


class Context:
    """All rows of one context up to depth gamma, and its draft blocks."""

    def __init__(self, cfg: dict, prefix: tuple[int, ...]):
        self.cfg = cfg
        self.vocab = cfg["vocab"]
        self.gamma = cfg["gamma"]
        self.num_drafts = cfg["drafts"]
        self.p: dict[tuple, list[float]] = {}
        self.q: dict[tuple, list[float]] = {}
        for depth in range(self.gamma + 1):
            for node in itertools.product(range(self.vocab), repeat=depth):
                p, q = node_distributions(
                    TEST_SALT, TEST_SEED, cfg["regime"], cfg["pair"], self.vocab,
                    cfg["temperature"], prefix + node,
                )
                self.p[node] = p
                self.q[node] = q
        self.blocks = list(itertools.product(range(self.vocab), repeat=self.gamma))
        self.block_prob = []
        self.block_json = []
        for block in self.blocks:
            prob = 1.0
            for t in range(self.gamma):
                prob *= self.q[block[:t]][block[t]]
            self.block_prob.append(prob)
            p_rows = [self.p[block[:t]] for t in range(self.gamma + 1)]
            q_rows = [self.q[block[:t]] for t in range(self.gamma)]
            self.block_json.append(
                (json.dumps(list(block)), json.dumps(p_rows), json.dumps(q_rows))
            )
        self.num_problems = len(self.blocks) ** self.num_drafts

    def problem(self, index: int) -> tuple[list[int], float]:
        base = len(self.blocks)
        picks = []
        weight = 1.0
        for _ in range(self.num_drafts):
            index, b = divmod(index, base)
            picks.append(b)
            weight *= self.block_prob[b]
        return picks, weight

    def request(self, rid: int, picks: list[int]) -> str:
        parts = [self.block_json[b] for b in picks]
        return (
            '{"id": %d, "vocab_size": %d, "gamma": %d, "num_drafts": %d, '
            '"drafts": [%s], "target_probs": [%s], "draft_probs": [%s]}\n'
            % (
                rid, self.vocab, self.gamma, self.num_drafts,
                ", ".join(x[0] for x in parts),
                ", ".join(x[1] for x in parts),
                ", ".join(x[2] for x in parts),
            )
        )

    def completion_tv(self, emitted: dict[tuple, float]) -> float:
        """TV between (emitted prefix + target continuation) and p, over gamma+1 tokens."""
        depth = self.gamma + 1
        total = 0.0
        stack = [((), 0.0, 1.0)]
        while stack:
            node, flow, ptarget = stack.pop()
            if len(node) == depth:
                total += abs(flow - ptarget)
                continue
            row = self.p[node]
            for y in range(self.vocab):
                child = node + (y,)
                stack.append((child, flow * row[y] + emitted.get(child, 0.0), ptarget * row[y]))
        return 0.5 * total


def build_instances() -> dict[str, list[Context]]:
    instances: dict[str, list[Context]] = {}
    for exp_name, configs in experiment_configs().items():
        contexts = []
        for cfg in configs:
            for prefix in sample_contexts(TEST_SALT, TEST_SEED, cfg, TEST_CONTEXTS_PER_CONFIG[exp_name]):
                contexts.append(Context(cfg, prefix))
        instances[exp_name] = contexts
    return instances


# --------------------------------------------------------------------------
# Fixed baseline (computed here, independent of any method artifact)
# --------------------------------------------------------------------------


def _residual(p, q):
    w = [max(a - b, 0.0) for a, b in zip(p, q)]
    s = sum(w)
    return [v / s for v in w] if s > 0.0 else list(p)


def _tokenwise_outcomes(ctx: Context, picks: list[int]) -> list[tuple[tuple, float]]:
    drafts = [ctx.blocks[b] for b in picks]
    gamma = ctx.gamma
    out: list[tuple[tuple, float]] = []

    def stage(k, tau, path, root, prob):
        while k < len(drafts) and drafts[k][:tau] != path:
            k += 1
        if k >= len(drafts):
            out.extend((path + (y,), prob * w) for y, w in enumerate(root) if w > 0.0)
            return
        d = drafts[k]
        alive = 1.0
        for t in range(tau, gamma):
            row = root if (t == tau and root is not None) else ctx.p[d[:t]]
            qrow = ctx.q[d[:t]]
            a = min(1.0, row[d[t]] / qrow[d[t]]) if qrow[d[t]] > 0.0 else 0.0
            if alive * (1.0 - a) > 0.0:
                stage(k + 1, t, d[:t], _residual(row, qrow), prob * alive * (1.0 - a))
            alive *= a
        if alive > 0.0:
            row = ctx.p[d]
            out.extend((d + (y,), prob * alive * w) for y, w in enumerate(row) if w > 0.0)

    stage(0, 0, (), None, 1.0)
    return out


def fixed_tokenwise_block_efficiency(contexts: list[Context]) -> float:
    values = []
    for ctx in contexts:
        be = 0.0
        for index in range(ctx.num_problems):
            picks, weight = ctx.problem(index)
            for emitted, prob in _tokenwise_outcomes(ctx, picks):
                be += weight * prob * len(emitted)
        values.append(be)
    return sum(values) / len(values)


# --------------------------------------------------------------------------
# Running one verifier program
# --------------------------------------------------------------------------


class VerifierFailure(Exception):
    pass


def _parse_outcomes(ctx: Context, picks: list[int], rid: int, line: str) -> dict[tuple, float]:
    try:
        msg = json.loads(line)
    except (json.JSONDecodeError, ValueError) as exc:
        raise VerifierFailure(f"unparseable response: {exc}") from None
    if not isinstance(msg, dict) or msg.get("id") != rid:
        raise VerifierFailure("response id does not match request id")
    outcomes = msg.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise VerifierFailure("response has no outcomes")
    merged: dict[tuple, float] = {}
    total = 0.0
    for item in outcomes:
        if not isinstance(item, list) or len(item) != 4:
            raise VerifierFailure("outcome must be [draft_index, tau, token, probability]")
        k, tau, y, prob = item
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (k, tau, y)):
            raise VerifierFailure("draft_index, tau and token must be integers")
        if not (0 <= k < ctx.num_drafts and 0 <= tau <= ctx.gamma and 0 <= y < ctx.vocab):
            raise VerifierFailure(f"outcome out of range: {item}")
        if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not math.isfinite(prob):
            raise VerifierFailure(f"non-finite probability: {item}")
        if prob < -1e-12:
            raise VerifierFailure(f"negative probability: {item}")
        prob = max(float(prob), 0.0)
        emitted = ctx.blocks[picks[k]][:tau] + (y,)
        merged[emitted] = merged.get(emitted, 0.0) + prob
        total += prob
    if abs(total - 1.0) > MASS_TOL:
        raise VerifierFailure(f"outcome probabilities sum to {total!r}, not 1")
    return merged


def run_verifier(verifier: Path, instances: dict[str, list[Context]]) -> dict:
    contexts = [(exp, ci) for exp in sorted(instances) for ci in range(len(instances[exp]))]
    refs = []
    for cid, (exp, ci) in enumerate(contexts):
        refs.extend((cid, index) for index in range(instances[exp][ci].num_problems))
    rng = random.Random(SHUFFLE_SEED)
    probes = rng.sample(range(len(refs)), max(1, int(PROBE_FRACTION * len(refs))))
    stream = [(r, False) for r in refs] + [(refs[i], True) for i in probes]
    rng.shuffle(stream)
    # A probe must come after the first occurrence of the same problem.
    seen: set = set()
    ordered = []
    deferred = []
    for ref, is_probe in stream:
        if is_probe and ref not in seen:
            deferred.append((ref, True))
            continue
        ordered.append((ref, is_probe))
        if not is_probe:
            seen.add(ref)
    ordered.extend(deferred)

    emitted = [dict() for _ in contexts]
    block_eff = [0.0 for _ in contexts]
    first_answer: dict = {}
    probe_set = {refs[i] for i in probes}

    workdir = tempfile.mkdtemp(prefix="verifier_")
    stderr_path = Path(workdir) / "stderr.log"
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": workdir, "PYTHONHASHSEED": "0"}
    start = time.time()
    with open(stderr_path, "w") as err:
        proc = subprocess.Popen(
            [sys.executable, "-I", str(verifier.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
            cwd=workdir, env=env, text=True, bufsize=1,
        )
        killer = threading.Timer(TIME_BUDGET_S, proc.kill)
        killer.start()
        failure = None
        try:
            for rid, ((cid, index), is_probe) in enumerate(ordered):
                exp, ci = contexts[cid]
                ctx = instances[exp][ci]
                picks, weight = ctx.problem(index)
                try:
                    proc.stdin.write(ctx.request(rid, picks))
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    raise VerifierFailure("verifier exited early or timed out") from None
                line = proc.stdout.readline()
                if not line:
                    raise VerifierFailure("verifier produced no response (crash or time budget exceeded)")
                merged = _parse_outcomes(ctx, picks, rid, line)
                key = (cid, index)
                if is_probe:
                    ref_ans = first_answer[key]
                    for s in set(ref_ans) | set(merged):
                        if abs(ref_ans.get(s, 0.0) - merged.get(s, 0.0)) > PROBE_TOL:
                            raise VerifierFailure("verifier answered the same problem differently")
                    continue
                if key in probe_set:
                    first_answer[key] = merged
                acc = emitted[cid]
                for s, prob in merged.items():
                    acc[s] = acc.get(s, 0.0) + weight * prob
                    block_eff[cid] += weight * prob * len(s)
        except VerifierFailure as exc:
            failure = str(exc)
        finally:
            killer.cancel()
            try:
                proc.stdin.close()
            except OSError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    elapsed = time.time() - start
    result = {"calls": len(ordered), "seconds": round(elapsed, 1), "failure": failure}
    if failure is not None:
        return result

    per_exp: dict[str, dict] = {}
    worst_tv = 0.0
    for cid, (exp, ci) in enumerate(contexts):
        ctx = instances[exp][ci]
        tv = ctx.completion_tv(emitted[cid])
        worst_tv = max(worst_tv, tv)
        entry = per_exp.setdefault(exp, {"be": [], "tv": [], "breakdown": {}})
        entry["be"].append(block_eff[cid])
        entry["tv"].append(tv)
        entry["breakdown"].setdefault(config_label(ctx.cfg), []).append(block_eff[cid])
    if not worst_tv <= LOSSLESS_TOL:
        result["failure"] = (
            f"not lossless: completion total variation {worst_tv:.3e} > {LOSSLESS_TOL:g}"
        )
    result["per_experiment"] = per_exp
    return result


# --------------------------------------------------------------------------
# Discovery and output
# --------------------------------------------------------------------------


def discover_methods(inputs: list[str]) -> dict[str, Path]:
    roots = [Path(x) for x in inputs] if inputs else [Path("checkpoints")]
    found: dict[str, Path] = {}
    for root in roots:
        if (root / "verifier.py").is_file():
            found[root.resolve().name] = root / "verifier.py"
            continue
        if root.is_dir():
            for sub in sorted(root.iterdir()):
                if (sub / "verifier.py").is_file():
                    found[sub.name] = sub / "verifier.py"
    return dict(sorted(found.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("methods", nargs="*", help="method dirs (with verifier.py) or roots of them")
    parser.add_argument("--output", default="scoring/scores_test.json")
    args = parser.parse_args()

    methods = discover_methods(args.methods)
    if not methods:
        raise SystemExit("No verifier programs found (expected checkpoints/<method>/verifier.py).")
    instances = build_instances()

    scores: dict = {
        "_meta": {
            "slice": SLICE,
            "lossless_tolerance": LOSSLESS_TOL,
            "stream_seed": SHUFFLE_SEED,
            "failed_score": FAILED_SCORE,
            "contexts": {exp: len(ctxs) for exp, ctxs in instances.items()},
            "problems": {exp: sum(c.num_problems for c in ctxs) for exp, ctxs in instances.items()},
            "methods": {},
        },
        "experiments": {},
    }
    for exp, ctxs in instances.items():
        scores["experiments"][exp] = {
            "results": {},
            "constraint": {},
            "breakdown": {},
            "fixed_baseline": {"tokenwise_block_efficiency": fixed_tokenwise_block_efficiency(ctxs)},
        }

    for name, verifier in methods.items():
        print(f"[{SLICE}] evaluating {name} ({verifier})", flush=True)
        res = run_verifier(verifier, instances)
        status = "failed" if res["failure"] else "ok"
        scores["_meta"]["methods"][name] = {
            "verifier": str(verifier),
            "status": status,
            "failure": res["failure"],
            "calls": res["calls"],
            "seconds": res["seconds"],
        }
        for exp in instances:
            block = scores["experiments"][exp]
            if status == "failed":
                block["results"][name] = {"block_efficiency": FAILED_SCORE}
                block["constraint"][name] = {"lossless": False}
                continue
            data = res["per_experiment"][exp]
            block["results"][name] = {"block_efficiency": sum(data["be"]) / len(data["be"])}
            block["constraint"][name] = {"lossless": True, "max_completion_tv": max(data["tv"])}
            block["breakdown"][name] = {k: sum(v) / len(v) for k, v in sorted(data["breakdown"].items())}
        print(f"[{SLICE}] {name}: {status} {res['failure'] or ''} ({res['seconds']}s)", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")
    summary = {
        exp: {m: v["block_efficiency"] for m, v in block["results"].items()}
        for exp, block in scores["experiments"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
