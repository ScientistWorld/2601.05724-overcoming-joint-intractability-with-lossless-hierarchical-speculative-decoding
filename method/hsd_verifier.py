"""Hierarchical Speculative Decoding (HSD) verifier program.

Implements the paper's verification procedures and reports, for every
verification problem it is given, the exact probability of each possible
verification outcome (instead of drawing one random outcome):

* single draft (K = 1): Algorithm 2 (HSD with capped branch resampling),
  i.e. capped prefix ratios (Definitions 4-5), capped branch divergences
  (Definition 6), the backward acceptance scan with Equation (19), and one
  resample from Equation (20) or a bonus target token when the whole draft
  is accepted;
* several drafts (K > 1): Algorithm 3 (HSD combined with recursive reject
  sampling with replacement): drafts are visited in order, a draft is only
  verified if it extends the currently accepted prefix, verification restarts
  from the accepted node with the previous capped residual installed as that
  node's target distribution, and the last residual is sampled at the end.

Reporting the outcome distribution is equivalent to running Algorithm 2/3 with
fresh uniform draws: P(tau = t) = h_t * prod_{s > t} (1 - h_s) for the
independent backward-scan draws, and the emitted next token follows the
residual (or bonus) distribution at the stopping node.

The program speaks the gym's verifier protocol (see ``briefing/evaluation.md``):
one JSON request per stdin line, one JSON response per stdout line.  It is
self-contained (standard library only) so the evaluator can run it in
isolation; ``scripts/method.sh`` copies it to ``checkpoints/hsd/verifier.py``.
"""

from __future__ import annotations

import json
import sys

EPS = 1e-300


def _normalized(weights: list[float]) -> list[float] | None:
    total = sum(weights)
    if total <= 0.0:
        return None
    return [w / total for w in weights]


def hsd_chain(
    target_rows: list[list[float]],
    draft_rows: list[list[float]],
    draft: list[int],
    root_target: list[float] | None = None,
):
    """Algorithm 2 on one draft chain, as an exact outcome distribution.

    ``target_rows[t]`` is p(. | X_{1:t}) for t = 0..gamma, ``draft_rows[t]``
    is q(. | X_{1:t}) for t = 0..gamma-1 and ``draft`` is X_{1:gamma}.
    ``root_target`` optionally replaces p(. | X_{1:0}) (Algorithm 3, line 28).

    Returns ``(outcomes, rows)`` where ``outcomes`` is a list of
    ``(tau, probability, next_token_distribution)`` and ``rows`` the target
    rows actually used (with the root replacement applied).
    """
    gamma = len(draft)
    rows = list(target_rows)
    if root_target is not None:
        rows[0] = root_target
    vocab = len(draft_rows[0]) if gamma else len(rows[0])

    # Joint probability ratios r(X_{1:i}) = p(X_{1:i}) / q(X_{1:i}).
    ratio = [1.0]
    for i in range(gamma):
        tok = draft[i]
        q_tok = draft_rows[i][tok]
        ratio.append(ratio[-1] * (rows[i][tok] / q_tok) if q_tok > 0.0 else 0.0)

    # Definition 4/5: for a candidate X_{1:t} y the maximum prefix ratio index
    # m ranges over prefixes X_{1:i}, 1 <= i <= t.  With M_t = max(1, max_i
    # r(X_{1:i})) the capped prefix ratio is r*(X_{1:t} y) = r(X_{1:t} y) / M_t
    # (= min{r(X_{1:m}), 1} * r(X_{m+1:t} y)).
    cap = [1.0]
    for i in range(1, gamma + 1):
        cap.append(max(cap[-1], ratio[i]))

    accept = [1.0] * (gamma + 1)
    residual: list[list[float] | None] = [None] * (gamma + 1)
    for t in range(gamma):
        scale = ratio[t] / cap[t]  # r(X_{1:t}) / M_t
        # Capped branch divergences (Definition 6) over Branch(X_{1:t}).
        # The common factor q(X_{1:t}) of every branch joint cancels in both
        # Equation (19) and Equation (20) and is therefore omitted.
        d_pq = 0.0
        d_qp = 0.0
        excess = []
        for y in range(vocab):
            q_y = draft_rows[t][y]
            r_star = scale * rows[t][y] / q_y if q_y > 0.0 else float("inf")
            if q_y > 0.0 and r_star <= 1.0:
                d_qp += (1.0 - r_star) * q_y
                excess.append(0.0)
            else:
                mass = scale * rows[t][y] - q_y  # q(y) (r* - 1), finite form
                d_pq += mass
                excess.append(mass)
        # Equation (19): h_t = D*(p,q) / D*(q,p)  (t < gamma).
        if t > 0:
            accept[t] = 1.0 if d_qp <= EPS else min(1.0, d_pq / d_qp)
        # Equation (20): capped branch resampling distribution at X_{1:t}.
        residual[t] = _normalized(excess) or list(rows[t])
    # h_gamma = min{r*(X_{1:gamma}), 1}.
    if gamma > 0:
        accept[gamma] = min(1.0, ratio[gamma] / cap[gamma - 1])

    # Backward scan (Algorithm 2, lines 2-11) with independent uniforms.
    outcomes = []
    survive = 1.0
    for t in range(gamma, 0, -1):
        prob = survive * accept[t]
        if t == gamma:
            outcomes.append((t, prob, list(rows[gamma])))  # bonus token
        else:
            outcomes.append((t, prob, residual[t]))
        survive *= 1.0 - accept[t]
    outcomes.append((0, survive, residual[0] if gamma > 0 else list(rows[0])))
    return outcomes, rows


def verify(problem: dict) -> list[list]:
    """Return ``[[draft_index, tau, token, probability], ...]``.

    The emitted tokens of an outcome are ``drafts[draft_index][:tau] + [token]``.
    """
    drafts = problem["drafts"]
    target = problem["target_probs"]
    draft_probs = problem["draft_probs"]
    num_drafts = len(drafts)
    gamma = len(drafts[0])
    merged: dict[tuple[int, int, int], float] = {}

    def emit(k: int, tau: int, dist: list[float], prob: float) -> None:
        for y, w in enumerate(dist):
            if w > 0.0 and prob > 0.0:
                key = (k, tau, y)
                merged[key] = merged.get(key, 0.0) + prob * w

    # Algorithm 3 (reduces to Algorithm 2 when there is a single draft).
    def stage(k: int, tau: int, path: tuple, root_target, prob: float, last_k: int) -> None:
        while k < num_drafts and tuple(drafts[k][:tau]) != path:
            k += 1  # line 21: skip drafts whose prefix does not match
        if k >= num_drafts:
            emit(last_k, tau, root_target, prob)  # final resample
            return
        outcomes, rows = hsd_chain(
            target[k][tau:], draft_probs[k][tau:], drafts[k][tau:], root_target
        )
        for t, p_t, dist in outcomes:
            if p_t <= 0.0:
                continue
            new_tau = tau + t
            if new_tau == gamma:
                emit(k, gamma, dist, prob * p_t)  # bonus token, stop
            else:
                # lines 27-29: the capped residual becomes the target at X_{1:tau}
                stage(k + 1, new_tau, tuple(drafts[k][:new_tau]), dist, prob * p_t, k)

    stage(0, 0, (), None, 1.0, 0)
    return [[k, tau, y, p] for (k, tau, y), p in sorted(merged.items())]


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        response = {"id": request["id"], "outcomes": verify(request)}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
