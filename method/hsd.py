"""Hierarchical Speculative Decoding verifier on finite AR distributions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data.finite_ar import (
    SCENARIOS,
    VOCAB,
    branch_joint,
    conditional,
    sample_from,
    sequence_probability,
)


EPS = 1e-12


def _prefix_prob(tokens: tuple[str, ...], kind: str, temperature: float) -> float:
    if not tokens:
        return 1.0
    return sequence_probability(tokens, kind, temperature)


def _continuation_prob(
    context: tuple[str, ...], tokens: tuple[str, ...], kind: str, temperature: float
) -> float:
    prob = 1.0
    running = context
    for tok in tokens:
        prob *= conditional(kind, running, temperature)[tok]
        running = running + (tok,)
    return prob


def _branch_joint_context(
    context: tuple[str, ...], prefix: tuple[str, ...], tok: str, kind: str, temperature: float
) -> float:
    return _continuation_prob(context, prefix, kind, temperature) * conditional(
        kind, context + prefix, temperature
    )[tok]


def _ratio(tokens: tuple[str, ...], temperature: float, context: tuple[str, ...] = ()) -> float:
    q = _continuation_prob(context, tokens, "draft", temperature)
    if q <= EPS:
        return 0.0
    return _continuation_prob(context, tokens, "target", temperature) / q


def max_prefix_ratio_index(
    tokens: tuple[str, ...], temperature: float, context: tuple[str, ...] = ()
) -> int:
    """Definition 4: argmax prefix index before the current position, or 0."""
    if len(tokens) <= 1:
        return 0
    ratios = [(_ratio(tokens[:i], temperature, context), i) for i in range(1, len(tokens))]
    best_ratio, best_i = max(ratios)
    return best_i if best_ratio > 1.0 else 0


def capped_ratio(
    tokens: tuple[str, ...], temperature: float, context: tuple[str, ...] = ()
) -> float:
    """Definition 5, using conditional suffix ratio after the capping point."""
    if not tokens:
        return 1.0
    m = max_prefix_ratio_index(tokens, temperature, context)
    if m == 0:
        return _ratio(tokens, temperature, context)
    prefix = tokens[:m]
    suffix = tokens[m:]
    q_suffix = 1.0
    p_suffix = 1.0
    running = context + prefix
    for tok in suffix:
        p_suffix *= conditional("target", running, temperature)[tok]
        q_suffix *= conditional("draft", running, temperature)[tok]
        running = running + (tok,)
    return p_suffix / max(q_suffix, EPS)


def capped_branch_divergence(
    prefix: tuple[str, ...],
    order: str,
    temperature: float,
    context: tuple[str, ...] = (),
) -> float:
    """Definitions 6: D*_Branch(p,q|prefix) or D*_Branch(q,p|prefix)."""
    total = 0.0
    for tok in VOCAB:
        candidate = prefix + (tok,)
        q_joint = _branch_joint_context(context, prefix, tok, "draft", temperature)
        r_star = capped_ratio(candidate, temperature, context)
        if order == "pq" and r_star > 1.0:
            total += (r_star - 1.0) * q_joint
        elif order == "qp" and r_star <= 1.0:
            total += (1.0 - r_star) * q_joint
    return total


def acceptance_probabilities(
    draft: tuple[str, ...], temperature: float, context: tuple[str, ...] = ()
) -> list[float]:
    """Equation 19 for t < gamma and h_gamma = min(r*(X_1:gamma), 1)."""
    gamma = len(draft)
    hs: list[float] = []
    for t in range(1, gamma + 1):
        prefix = draft[:t]
        if t == gamma:
            h = min(capped_ratio(prefix, temperature, context), 1.0)
        else:
            d_pq = capped_branch_divergence(prefix, "pq", temperature, context)
            d_qp = capped_branch_divergence(prefix, "qp", temperature, context)
            if d_qp <= EPS:
                h = 1.0
            else:
                h = min(1.0, d_pq / d_qp)
        hs.append(max(0.0, min(1.0, h)))
    return hs


def resample_capped(
    prefix: tuple[str, ...],
    rng: random.Random,
    temperature: float,
    context: tuple[str, ...] = (),
) -> str:
    """Equation 20: sample from P*_res(. | prefix)."""
    weights = {}
    denom = capped_branch_divergence(prefix, "pq", temperature, context)
    if denom <= EPS:
        return sample_from(conditional("target", context + prefix, temperature), rng)
    for tok in VOCAB:
        candidate = prefix + (tok,)
        q_joint = _branch_joint_context(context, prefix, tok, "draft", temperature)
        weights[tok] = max(q_joint * (capped_ratio(candidate, temperature, context) - 1.0), 0.0) / denom
    total = sum(weights.values())
    if total <= EPS:
        return sample_from(conditional("target", context + prefix, temperature), rng)
    weights = {tok: val / total for tok, val in weights.items()}
    return sample_from(weights, rng)


def draft_block(prefix: tuple[str, ...], gamma: int, rng: random.Random, temperature: float) -> tuple[str, ...]:
    out = []
    running = prefix
    for _ in range(gamma):
        tok = sample_from(conditional("draft", running, temperature), rng)
        out.append(tok)
        running = running + (tok,)
    return tuple(out)


def hsd_step(prefix: tuple[str, ...], gamma: int, rng: random.Random, temperature: float) -> tuple[tuple[str, ...], int]:
    """Algorithm 2: backward scan, accept longest suffix-confirmed prefix, one resample."""
    draft = draft_block(prefix, gamma, rng, temperature)
    hs = acceptance_probabilities(draft, temperature, prefix)
    tau = 0
    for idx in range(gamma, 0, -1):
        if rng.random() <= hs[idx - 1]:
            tau = idx
            break
    accepted = draft[:tau]
    new_prefix = prefix + accepted
    if tau == gamma:
        token = sample_from(conditional("target", new_prefix, temperature), rng)
    else:
        token = resample_capped(accepted, rng, temperature, prefix)
    return accepted + (token,), tau + 1


def generate(max_len: int, gamma: int, seed: int, temperature: float) -> tuple[str, int, int]:
    rng = random.Random(seed)
    prefix: tuple[str, ...] = ()
    target_steps = 0
    emitted = 0
    while len(prefix) < max_len:
        block, block_eff = hsd_step(prefix, min(gamma, max_len - len(prefix)), rng, temperature)
        remaining = max_len - len(prefix)
        prefix = prefix + block[:remaining]
        emitted += min(block_eff, remaining)
        target_steps += 1
    return "".join(prefix), emitted, target_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="finite_ar_main", choices=sorted(SCENARIOS))
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="checkpoints/hsd_finite_ar_main.json")
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]
    counts: dict[str, int] = {}
    emitted_total = 0
    step_total = 0
    for i in range(args.samples):
        seq, emitted, steps = generate(
            scenario.max_len, scenario.gamma, args.seed + i, scenario.temperature
        )
        counts[seq] = counts.get(seq, 0) + 1
        emitted_total += emitted
        step_total += steps

    artifact = {
        "method": "hsd",
        "scenario": scenario.name,
        "gamma": scenario.gamma,
        "max_len": scenario.max_len,
        "temperature": scenario.temperature,
        "samples": args.samples,
        "sequence_counts": counts,
        "generated_tokens": emitted_total,
        "target_steps": step_total,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
