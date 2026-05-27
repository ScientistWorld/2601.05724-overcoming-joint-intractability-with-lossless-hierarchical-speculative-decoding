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
CNN/DailyMail; this workspace uses those reported block-efficiency values as
reference targets while running a finite scaled benchmark.

## Evaluation Protocol

Run `scripts/reproduce.sh` to generate baseline and candidate artifacts, then
run `scripts/evaluate.sh`. Evaluation reads JSON artifacts from `checkpoints/`
and writes `scoring/scores.json`. Candidate methods must use the same finite
target/draft distributions, sequence length, temperature, and draft length for
each named experiment.
