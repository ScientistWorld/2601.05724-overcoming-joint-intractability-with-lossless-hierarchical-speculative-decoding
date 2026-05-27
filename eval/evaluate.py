"""Evaluate speculative decoding artifacts without importing method code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.finite_ar import SCENARIOS, exact_sequence_distribution


def total_variation(counts: dict[str, int], samples: int, max_len: int, temperature: float) -> float:
    exact = exact_sequence_distribution(max_len, temperature)
    keys = set(exact) | set(counts)
    return 0.5 * sum(abs(counts.get(key, 0) / samples - exact.get(key, 0.0)) for key in keys)


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
    block_efficiency = float(data["generated_tokens"]) / float(data["target_steps"])
    tv = total_variation(counts, samples, scenario.max_len, scenario.temperature)
    return {
        "block_efficiency": block_efficiency,
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
