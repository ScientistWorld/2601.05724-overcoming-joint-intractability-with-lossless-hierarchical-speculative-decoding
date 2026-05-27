"""Held-out test-slice scoring for finite AR speculative decoding artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data.finite_ar import SCENARIOS, VOCAB, exact_sequence_distribution


SLICE = "test"
SPLIT_SALT = "finite-ar-heldout-test-v1"


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
        raise ValueError(f"{scenario_name}: test split selected no sequences")

    empirical_mass = sum(counts.get(seq, 0) for seq in selected) / samples
    exact_mass = sum(exact[seq] for seq in selected)
    slice_tv = 0.5 * sum(abs(counts.get(seq, 0) / samples - exact[seq]) for seq in selected)
    return {
        "slice_total_variation": slice_tv,
        "slice_mass_error": abs(empirical_mass - exact_mass),
    }


def _prefix_member(scenario_name: str, prefix: str) -> bool:
    key = f"{SPLIT_SALT}:prefix:{scenario_name}:{prefix}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") % 2 == 0


def _prefix_conditional_l1(
    scenario_name: str,
    counts: dict[str, int],
    max_len: int,
    temperature: float,
) -> float:
    exact = exact_sequence_distribution(max_len, temperature)
    prefixes = sorted(
        {
            seq[:length]
            for seq in exact
            for length in range(max_len)
            if _prefix_member(scenario_name, seq[:length])
        }
    )
    if not prefixes:
        raise ValueError(f"{scenario_name}: test split selected no prefixes")

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
        "prefix_conditional_l1": _prefix_conditional_l1(
            scenario_name, counts, scenario.max_len, scenario.temperature
        ),
    }
    metrics.update(
        _slice_metrics(scenario_name, counts, samples, scenario.max_len, scenario.temperature)
    )
    return scenario_name, str(data["method"]), metrics


def _read_artifact_identity(path: Path) -> tuple[str, str]:
    data = json.loads(path.read_text())
    return str(data["scenario"]), str(data["method"])


def _discover_artifacts(paths: list[str]) -> dict[tuple[str, str], Path]:
    candidates = [Path(name) for name in paths] if paths else sorted(Path("checkpoints").glob("*.json"))
    found: dict[tuple[str, str], Path] = {}
    for artifact in candidates:
        found[_read_artifact_identity(artifact)] = artifact
    return found


def _required_from_train(train_scores: Path) -> tuple[dict[tuple[str, str], Path], set[tuple[str, str]]]:
    if not train_scores.exists():
        return {}, set()

    data = json.loads(train_scores.read_text())
    meta_paths: dict[tuple[str, str], Path] = {}
    for exp_name, methods in data.get("_meta", {}).get("artifacts", {}).items():
        for method_name, artifact_name in methods.items():
            meta_paths[(str(exp_name), str(method_name))] = Path(artifact_name)

    required: set[tuple[str, str]] = set()
    for exp_name, exp_data in data.get("experiments", {}).items():
        if str(exp_name).startswith("_"):
            continue
        for method_name in exp_data.get("results", {}):
            if str(method_name).startswith("_"):
                continue
            required.add((str(exp_name), str(method_name)))
    return meta_paths, required


def _artifact_paths(paths: list[str], train_scores: Path) -> list[Path]:
    discovered = _discover_artifacts(paths)
    meta_paths, required = _required_from_train(train_scores)

    if not required:
        if not discovered:
            raise SystemExit("No checkpoint artifacts found. Pass artifact paths or run scripts/method.sh.")
        return [discovered[key] for key in sorted(discovered)]

    selected: list[Path] = []
    for key in sorted(required):
        artifact = discovered.get(key) or meta_paths.get(key)
        if artifact is None or not artifact.exists():
            exp_name, method_name = key
            raise SystemExit(
                f"Missing checkpoint artifact for method {method_name!r} in experiment {exp_name!r}"
            )
        selected.append(artifact)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", nargs="*", default=[])
    parser.add_argument("--train-scores", default="scoring/scores_train.json")
    parser.add_argument("--output", default="scoring/scores_test.json")
    args = parser.parse_args()

    scores: dict[str, object] = {
        "_meta": {"slice": SLICE, "artifacts": {}, "matched_train_scores": args.train_scores},
        "experiments": {},
    }
    artifact_meta: dict[str, dict[str, str]] = scores["_meta"]["artifacts"]  # type: ignore[index]

    for artifact in _artifact_paths(args.artifacts, Path(args.train_scores)):
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
