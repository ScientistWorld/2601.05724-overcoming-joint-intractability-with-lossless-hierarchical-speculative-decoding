"""Finite-state calculation to check the actual reproduced HSD and baseline.

This audit diagnostic uses the method; candidate generators never use this file.
Every draft and verification outcome is integrated, rather than sampled.
"""

from collections import defaultdict
import itertools
import json
from pathlib import Path

from baseline import tokenwise
from data import finite_ar as data
from method import hsd
from redteam.generate import cache_pure_functions


def enumerate_outputs(scenario, kind):
    state = defaultdict(float, {(): 1.0})
    outputs = defaultdict(float)
    expected_steps = 0.0
    for length in range(scenario.max_len):
        for context in itertools.product(data.VOCAB, repeat=length):
            weight = state[context]
            if not weight:
                continue
            expected_steps += weight
            gamma = min(scenario.gamma, scenario.max_len - length)
            for draft in itertools.product(data.VOCAB, repeat=gamma):
                draft_prob = hsd._continuation_prob(context, draft, "draft", scenario.temperature)
                if kind == "hsd":
                    hs = hsd.acceptance_probabilities(draft, scenario.temperature, context)
                    remaining = 1.0
                    tau_probs = {}
                    for tau in range(gamma, 0, -1):
                        tau_probs[tau] = remaining * hs[tau - 1]
                        remaining *= 1.0 - hs[tau - 1]
                    tau_probs[0] = remaining
                else:
                    remaining = 1.0
                    tau_probs = {}
                    for tau, tok in enumerate(draft):
                        prefix = context + draft[:tau]
                        p = data.conditional("target", prefix, scenario.temperature)[tok]
                        q = data.conditional("draft", prefix, scenario.temperature)[tok]
                        accept = min(1.0, p / q)
                        tau_probs[tau] = remaining * (1.0 - accept)
                        remaining *= accept
                    tau_probs[gamma] = remaining
                for tau, tau_prob in tau_probs.items():
                    mass = weight * draft_prob * tau_prob
                    prefix = context + draft[:tau]
                    if not mass:
                        continue
                    if len(prefix) >= scenario.max_len:
                        outputs["".join(prefix[:scenario.max_len])] += mass
                        continue
                    if tau == gamma:
                        probabilities = data.conditional("target", prefix, scenario.temperature)
                    elif kind == "hsd":
                        probabilities = {}
                        for tok in data.VOCAB:
                            candidate = draft[:tau] + (tok,)
                            q = hsd._branch_joint_context(context, draft[:tau], tok,
                                                         "draft", scenario.temperature)
                            probabilities[tok] = max(0.0, q * (
                                hsd.capped_ratio(candidate, scenario.temperature, context) - 1.0))
                    else:
                        p = data.conditional("target", prefix, scenario.temperature)
                        q = data.conditional("draft", prefix, scenario.temperature)
                        probabilities = {tok: max(0.0, p[tok] - q[tok]) for tok in data.VOCAB}
                    total = sum(probabilities.values())
                    if total <= 1e-12:
                        probabilities = data.conditional("target", prefix, scenario.temperature)
                        total = 1.0
                    for tok, probability in probabilities.items():
                        extended = prefix + (tok,)
                        part = mass * probability / total
                        if len(extended) == scenario.max_len:
                            outputs["".join(extended)] += part
                        else:
                            state[extended] += part
    exact = data.exact_sequence_distribution(scenario.max_len, scenario.temperature)
    return {"total_variation": 0.5 * sum(abs(outputs[key] - value) for key, value in exact.items()),
            "probability_mass": sum(outputs.values()),
            "expected_target_steps_per_sequence": expected_steps,
            "raw_block_efficiency": scenario.max_len / expected_steps}


def main():
    for module in (data, hsd, tokenwise):
        cache_pure_functions(module)
    results = {}
    for scenario in data.SCENARIOS.values():
        results[scenario.name] = {kind: enumerate_outputs(scenario, kind)
                                  for kind in ("hsd", "tokenwise")}
    out = Path("redteam/reference/exact_verification.json")
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
