"""Rank attempts the way the gym ranks them (hardening report helper).

rank(attempt) = sum_exp weight_exp * sum_metric |coef| * dir * (x - x_ref) / std_pop(x)

x_ref is the frozen reference (method ``hsd`` in scoring/scores_test.json),
dir = +1 for higher_is_better, std_pop is the population standard deviation
of the metric over the attempts being ranked.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = Path(__file__).resolve().parent / "results"


def load_attempts():
    attempts = {}
    sources = [
        (ROOT / "scoring" / "scores_test.json", ""),
        (RES / "hsd_rerun.json", "rerun:"),
        (RES / "variants.json", "variant:"),
        (RES / "shortcuts.json", "shortcut:"),
    ]
    for path, prefix in sources:
        data = json.loads(path.read_text())
        for exp, block in data["experiments"].items():
            for method, metrics in block["results"].items():
                attempts.setdefault(prefix + method, {})[exp] = metrics
    return attempts


def rank(population, attempts, ref, spec):
    out = {name: 0.0 for name in population}
    for exp, exp_spec in spec.items():
        for metric, mdef in exp_spec["metrics"].items():
            coef = abs(mdef["coefficient"])
            if coef == 0:
                continue
            direction = 1.0 if mdef["higher_is_better"] else -1.0
            vals = [attempts[n][exp][metric] for n in population]
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            if std == 0:
                continue
            for n in population:
                out[n] += exp_spec["weight"] * coef * direction * (attempts[n][exp][metric] - ref[exp][metric]) / std
    return out


def main() -> None:
    spec = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
    attempts = load_attempts()
    ref = attempts["hsd"]
    honest = [n for n in attempts if not n.startswith("shortcut:")]
    everyone = list(attempts)
    report = {
        "honest_population": rank(honest, attempts, ref, spec),
        "all_population": rank(everyone, attempts, ref, spec),
        "block_efficiency": {n: {e: attempts[n][e]["block_efficiency"] for e in spec} for n in everyone},
    }
    RES.mkdir(exist_ok=True)
    (RES / "rank.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    exps = list(spec)
    lines = ["| attempt | " + " | ".join(exps) + " | rank (honest pop.) | rank (all) |",
             "|---|" + "---:|" * (len(exps) + 2)]
    for n in sorted(everyone, key=lambda n: -report["all_population"][n]):
        hon = report["honest_population"].get(n)
        lines.append(
            f"| {n} | " + " | ".join(f"{attempts[n][e]['block_efficiency']:.4f}" for e in exps)
            + f" | {'' if hon is None else f'{hon:+.4f}'} | {report['all_population'][n]:+.4f} |"
        )
    (RES / "rank.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
