# Evaluation

## Metrics

- `block_efficiency`: generated tokens per serial target-model step. Higher is
  better.
- `total_variation`: empirical total-variation distance between generated
  sequence counts and the exact target distribution. Lower is better.

## Baselines and Targets

The primary baseline is standard tokenwise speculative decoding. This workspace
evaluates a finite scaled benchmark with the same general objective as the
paper: improve block efficiency over tokenwise decoding while preserving the
exact target distribution. The original large-model paper numbers are
summarized outside the method-agnostic problem definition; `scoring/reference.json`
contains the finite benchmark reference values used for automated comparison.

## Evaluation Protocol

Run `scripts/reproduce.sh` to generate baseline and candidate artifacts, then
run `scripts/evaluate.sh`. Evaluation reads JSON artifacts from `checkpoints/`
and writes `scoring/scores.json`. Candidate methods must use the same finite
target/draft distributions, sequence length, temperature, and draft length for
each named experiment.
