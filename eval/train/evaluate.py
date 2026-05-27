"""Train-slice scoring for finite AR speculative decoding artifacts.

This evaluator is intentionally method-agnostic: it reads checkpoint JSON files
produced by any method that follows the repository artifact contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data.finite_ar import SCENARIOS, exact_sequence_distribution


SLICE = "train"
SPLIT_SALT = "finite-ar-visible-train-v1"


def _support_member(scenario_name: str, sequence: str) -> bool:
    key = f"{SPLIT_SALT}:{scenario_name}:{sequence}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") % 2 == 0


def _total_variation(counts: dict[str, int], samples: int, max_len: int, temperature: float) -> float:
    exact = exact_sequence_distribution(max_len, temperature)
    keys = set(exact) | set(counts)
    return 0.5 * sum(abs(counts.get(key, 0) / samples - exact.get(key, 0.0)) for key in keys)


def _slice_metrics(
    scenario_name: str,
    counts: dict[str, int],
    samples: int,
    max_len: int,
    temperature: float,
) -> dict[str, float]:
    exact = exact_sequence_distribution(max_len, temperature)
    selected = [seq for seq in exact if _support_member(scenario_name, seq)]
    if not selected:
        raise ValueError(f"{scenario_name}: train split selected no sequences")

    empirical_mass = sum(counts.get(seq, 0) for seq in selected) / samples
    exact_mass = sum(exact[seq] for seq in selected)
    slice_tv = 0.5 * sum(abs(counts.get(seq, 0) / samples - exact[seq]) for seq in selected)
    return {
        "slice_total_variation": slice_tv,
        "slice_mass_error": abs(empirical_mass - exact_mass),
    }


def evaluate_artifact(path: Path) -> tuple[str, str, dict[str, float]]:
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
    metrics = {
        "block_efficiency": block_efficiency,
        "total_variation": _total_variation(counts, samples, scenario.max_len, scenario.temperature),
    }
    metrics.update(
        _slice_metrics(scenario_name, counts, samples, scenario.max_len, scenario.temperature)
    )
    return scenario_name, str(data["method"]), metrics


def _artifact_paths(paths: list[str]) -> list[Path]:
    if paths:
        return [Path(name) for name in paths]
    artifacts = sorted(Path("checkpoints").glob("*.json"))
    if not artifacts:
        raise SystemExit("No checkpoint artifacts found. Pass artifact paths or run scripts/method.sh.")
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", nargs="*", default=[])
    parser.add_argument("--output", default="scoring/scores_train.json")
    args = parser.parse_args()

    scores: dict[str, object] = {
        "_meta": {"slice": SLICE, "artifacts": {}},
        "experiments": {},
    }
    artifact_meta: dict[str, dict[str, str]] = scores["_meta"]["artifacts"]  # type: ignore[index]

    for artifact in _artifact_paths(args.artifacts):
        exp_name, method_name, metrics = evaluate_artifact(artifact)
        experiments = scores["experiments"]  # type: ignore[assignment]
        experiments.setdefault(exp_name, {"results": {}})
        experiments[exp_name]["results"][method_name] = metrics
        artifact_meta.setdefault(exp_name, {})[method_name] = str(artifact)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, sort_keys=True))
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
