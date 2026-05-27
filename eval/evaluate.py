"""Evaluate speculative decoding artifacts without importing method code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.finite_ar import SCENARIOS, VOCAB, exact_sequence_distribution


def total_variation(counts: dict[str, int], samples: int, max_len: int, temperature: float) -> float:
    exact = exact_sequence_distribution(max_len, temperature)
    keys = set(exact) | set(counts)
    return 0.5 * sum(abs(counts.get(key, 0) / samples - exact.get(key, 0.0)) for key in keys)


def prefix_conditional_l1(counts: dict[str, int], max_len: int, temperature: float) -> float:
    exact = exact_sequence_distribution(max_len, temperature)
    prefixes = sorted({seq[:length] for seq in exact for length in range(max_len)})
    weighted_l1 = 0.0
    total_weight = 0.0

    for prefix in prefixes:
        exact_prefix_mass = sum(prob for seq, prob in exact.items() if seq.startswith(prefix))
        empirical_prefix_count = sum(count for seq, count in counts.items() if seq.startswith(prefix))
        if exact_prefix_mass <= 0.0:
            continue

        next_index = len(prefix)
        l1 = 0.0
        for tok in VOCAB:
            exact_next_mass = sum(
                prob
                for seq, prob in exact.items()
                if seq.startswith(prefix) and seq[next_index] == tok
            )
            empirical_next_count = sum(
                count
                for seq, count in counts.items()
                if seq.startswith(prefix) and seq[next_index] == tok
            )
            exact_cond = exact_next_mass / exact_prefix_mass
            empirical_cond = (
                empirical_next_count / empirical_prefix_count
                if empirical_prefix_count > 0
                else 0.0
            )
            l1 += abs(empirical_cond - exact_cond)

        weighted_l1 += exact_prefix_mass * l1
        total_weight += exact_prefix_mass

    return weighted_l1 / total_weight


def evaluate_artifact(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    scenario_name = str(data["scenario"])
    if scenario_name not in SCENARIOS:
        raise ValueError(f"{path}: unknown scenario {scenario_name!r}")
    scenario = SCENARIOS[scenario_name]
    if int(data["gamma"]) != scenario.gamma:
        raise ValueError(f"{path}: gamma does not match fixed scenario {scenario_name}")
    if int(data["max_len"]) != scenario.max_len:
        raise ValueError(f"{path}: max_len does not match fixed scenario {scenario_name}")
    if abs(float(data["temperature"]) - scenario.temperature) > 1e-12:
        raise ValueError(f"{path}: temperature does not match fixed scenario {scenario_name}")
    samples = int(data["samples"])
    counts = {str(k): int(v) for k, v in data["sequence_counts"].items()}
    observed = sum(counts.values())
    if observed != samples:
        raise ValueError(f"{path}: sequence_counts sum to {observed}, expected {samples}")

    block_efficiency = float(data["generated_tokens"]) / float(data["target_steps"])
    tv = total_variation(counts, samples, scenario.max_len, scenario.temperature)
    return {
        "block_efficiency": block_efficiency,
        "prefix_conditional_l1": prefix_conditional_l1(
            counts, scenario.max_len, scenario.temperature
        ),
        "slice_block_efficiency": block_efficiency,
        "slice_mass_error": 0.0,
        "slice_total_variation": tv,
        "total_variation": tv,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", nargs="+", required=True)
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    scores = {"experiments": {}}
    for artifact_name in args.artifacts:
        artifact = Path(artifact_name)
        data = json.loads(artifact.read_text())
        exp_name = data["scenario"]
        method_name = data["method"]
        scores["experiments"].setdefault(exp_name, {"results": {}})
        scores["experiments"][exp_name]["results"][method_name] = evaluate_artifact(artifact)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, sort_keys=True))
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
