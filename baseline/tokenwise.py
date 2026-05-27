"""Token-wise speculative decoding baseline from Leviathan et al. (2023)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data.finite_ar import SCENARIOS, conditional, sample_from


EPS = 1e-12


def draft_block(prefix: tuple[str, ...], gamma: int, rng: random.Random, temperature: float) -> tuple[str, ...]:
    out = []
    running = prefix
    for _ in range(gamma):
        tok = sample_from(conditional("draft", running, temperature), rng)
        out.append(tok)
        running = running + (tok,)
    return tuple(out)


def residual_sample(prefix: tuple[str, ...], rng: random.Random, temperature: float) -> str:
    p = conditional("target", prefix, temperature)
    q = conditional("draft", prefix, temperature)
    weights = {tok: max(p[tok] - q[tok], 0.0) for tok in p}
    total = sum(weights.values())
    if total <= EPS:
        return sample_from(p, rng)
    return sample_from({tok: val / total for tok, val in weights.items()}, rng)


def tokenwise_step(prefix: tuple[str, ...], gamma: int, rng: random.Random, temperature: float) -> tuple[tuple[str, ...], int]:
    draft = draft_block(prefix, gamma, rng, temperature)
    accepted = []
    running = prefix
    for tok in draft:
        p = conditional("target", running, temperature)[tok]
        q = conditional("draft", running, temperature)[tok]
        if rng.random() <= min(1.0, p / max(q, EPS)):
            accepted.append(tok)
            running = running + (tok,)
        else:
            return tuple(accepted + [residual_sample(running, rng, temperature)]), len(accepted) + 1
    return tuple(accepted + [sample_from(conditional("target", running, temperature), rng)]), len(accepted) + 1


def generate(max_len: int, gamma: int, seed: int, temperature: float) -> tuple[str, int, int]:
    rng = random.Random(seed)
    prefix: tuple[str, ...] = ()
    target_steps = 0
    emitted = 0
    while len(prefix) < max_len:
        block, block_eff = tokenwise_step(prefix, min(gamma, max_len - len(prefix)), rng, temperature)
        remaining = max_len - len(prefix)
        prefix = prefix + block[:remaining]
        emitted += min(block_eff, remaining)
        target_steps += 1
    return "".join(prefix), emitted, target_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="finite_ar_main", choices=sorted(SCENARIOS))
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", default="checkpoints/tokenwise_finite_ar_main.json")
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
        "method": "tokenwise",
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
