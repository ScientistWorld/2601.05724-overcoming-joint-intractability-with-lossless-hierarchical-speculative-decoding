"""Apply the stated normalized ranking; never add metrics in their raw units."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "redteam"


def load(path):
    return json.loads(path.read_text())


def candidate(scores, method="hsd"):
    return {name: entry["results"][method]
            for name, entry in scores["experiments"].items()}


def main():
    reference = candidate(load(HERE / "reference/scores_test.json"))
    weights = load(HERE / "reference/weights.json")["experiments"]
    results = {}
    rejected = {}
    for name in load(HERE / "cases.json"):
        outcome_path = HERE / name / "outcome.json"
        if not outcome_path.exists():
            continue
        outcome = load(outcome_path)
        path = HERE / name / "scores_test.json"
        if outcome.get("test_exit") != 0 or not path.exists():
            rejected[name] = {"reason": "hidden evaluation failed", "outcome": outcome}
            continue
        value = candidate(load(path))
        if not all(math.isfinite(v) for exp in value.values() for v in exp.values()):
            rejected[name] = {"reason": "evaluator accepted nonfinite scores; finite rank undefined"}
            continue
        results[name] = value

    honest_names = ["reference_seed1", "reference_seed2", "reference_seed3",
                    "better_more_samples", "weaker_tokenwise", "weaker_fewer_samples"]
    failure_names = honest_names + ["draft_only_shortcut", "constant_shortcut"]
    cohorts = {"honest": honest_names, "all_finite": list(results),
               "task_failure": failure_names}
    ranked = {}
    for label, names in cohorts.items():
        names = [name for name in names if name in results]
        scales = {}
        for experiment, definition in weights.items():
            scales[experiment] = {}
            for metric in definition["metrics"]:
                values = [results[name][experiment][metric] for name in names]
                scales[experiment][metric] = statistics.pstdev(values)

        ranks, contributions = {}, {}
        for name in names:
            total = 0.0
            terms = {}
            for experiment, definition in weights.items():
                terms[experiment] = {}
                for metric, spec in definition["metrics"].items():
                    sigma = scales[experiment][metric]
                    direction = 1.0 if spec["higher_is_better"] else -1.0
                    delta = direction * (results[name][experiment][metric] - reference[experiment][metric])
                    term = (definition["weight"] * abs(spec["coefficient"]) * delta / sigma
                            if sigma > 0 else 0.0)
                    terms[experiment][metric] = term
                    total += term
            ranks[name] = total
            contributions[name] = terms
        ranked[label] = {"attempts": names, "population_standard_deviations": scales,
                         "rank_improvement": ranks, "metric_contributions": contributions}

    seeds = ["reference_seed1", "reference_seed2", "reference_seed3"]
    seed_spread = {}
    for experiment, definition in weights.items():
        seed_spread[experiment] = {}
        for metric in definition["metrics"]:
            values = [results[name][experiment][metric] for name in seeds]
            seed_spread[experiment][metric] = {
                "values": values, "min": min(values), "max": max(values),
                "range": max(values) - min(values), "population_std": statistics.pstdev(values)}
    scalar_spread = {}
    for label, cohort in ranked.items():
        values = [cohort["rank_improvement"][name] for name in seeds]
        scalar_spread[label] = {"values": values, "range": max(values) - min(values)}

    weighted = {}
    for name, experiments in results.items():
        weighted[name] = {metric: sum(weights[experiment]["weight"] * values[metric]
                                      for experiment, values in experiments.items())
                          for metric in next(iter(weights.values()))["metrics"]}
    dominance = {}
    for name, experiments in results.items():
        better = worse = equal = 0
        for experiment, values in experiments.items():
            for metric, value in values.items():
                direction = 1 if weights[experiment]["metrics"][metric]["higher_is_better"] else -1
                delta = direction * (value - reference[experiment][metric])
                better += delta > 1e-14
                worse += delta < -1e-14
                equal += abs(delta) <= 1e-14
        dominance[name] = {"strictly_better_metrics": better, "strictly_worse_metrics": worse,
                           "equal_metrics": equal}

    summary = {"reference": reference, "reference_rank_improvement": 0.0,
        "ranking_convention": "sum weight * abs(coefficient) * oriented_delta / population_std; "
            "std over explicitly listed finite attempts, not over metric units. "
            "Unknown future attempt population can change these scalar ranks.",
        "cohorts": ranked, "seed_metric_spread": seed_spread,
        "seed_rank_spread": scalar_spread, "weighted_metric_means": weighted,
        "dominance_over_reference": dominance, "rejected_or_undefined": rejected}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for label, cohort in ranked.items():
        print(label)
        for name, value in sorted(cohort["rank_improvement"].items(), key=lambda x: -x[1]):
            print(f"  {name:30s} {value:+.6f}")
    print("seed rank ranges", scalar_spread)
    print("rejected or undefined", list(rejected))


if __name__ == "__main__":
    main()
