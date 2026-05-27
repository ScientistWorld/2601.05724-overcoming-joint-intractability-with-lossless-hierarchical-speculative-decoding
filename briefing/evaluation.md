# Evaluation

How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results. Think of this as a general evaluation protocal for other scientists trying to solve the same problem, where they should not see this paper's solution.

## Metrics

- `block_efficiency`: generated tokens per serial target-model step. Higher is
  better.
- `total_variation`: empirical total-variation distance between generated
  sequence counts and the exact target distribution. Lower is better.

## Baselines and Targets

The primary baseline is tokenwise speculative decoding. The paper reports HSD
improvements over tokenwise block efficiency across GSM8K, HumanEval, and
CNN/DailyMail. This workspace evaluates a finite scaled benchmark with the same
verifier objective: improve block efficiency over tokenwise decoding while
preserving the exact target distribution. The original large-model paper
numbers are summarized in `briefing/overview.md`; `scoring/reference.json`
contains the finite benchmark reference values used for automated comparison.

## Evaluation Protocol

Run `scripts/reproduce.sh` to generate baseline and candidate artifacts, then
run `scripts/evaluate.sh`. Evaluation reads JSON artifacts from `checkpoints/`
and writes `scoring/scores.json`. Candidate methods must use the same finite
target/draft distributions, sequence length, temperature, and draft length for
each named experiment.
