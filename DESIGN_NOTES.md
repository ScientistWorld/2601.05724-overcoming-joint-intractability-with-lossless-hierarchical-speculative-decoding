# Train/Test Evaluation Design Notes

## Checks performed
- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. The train script invokes `eval.train.evaluate` and writes `scoring/scores_train.json`; the test script invokes `eval.test.evaluate` and writes `scoring/scores_test.json`.
- Checked for hardcoded method defaults in the split evaluators. Both scripts enumerate JSON artifacts under `checkpoints/` when no paths are supplied and also accept explicit artifact paths.
- Checked for train/test leakage by grepping for cross-imports between `eval/train/` and `eval/test/`. No cross-imports were found.
- Confirmed the evaluators read method artifacts (`sequence_counts`, `generated_tokens`, `target_steps`) rather than importing or re-running `method/`.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` do not exist.
- Ran `bash -n` on both split evaluator scripts.

## Changes made
- Added `prefix_conditional_l1` to both split evaluators. This metric compares next-token conditional distributions on split-specific hashed prefix subsets, weighted by exact target prefix mass.
- Added the new metric to all existing experiments in `scoring/reference.json` with negative coefficient.
- Added missing `description` fields for the existing `block_efficiency` and `total_variation` metric entries in `scoring/reference.json`.

## Design rationale
- The first pass already included global benefit and constraint metrics plus split-specific sequence-support metrics. The main weakness was that the only split-sensitive checks were support mass and support TV; a shortcut could still be tuned to visible sequence subsets while hiding local autoregressive distortions.
- `prefix_conditional_l1` adds a different view of the same scientific constraint: samples should match the target autoregressive process locally, not only in aggregate sequence mass. A method that overfits visible train support now has another hidden axis where it can regress.
- The metric is computed from existing artifacts, so test evaluation remains lightweight and method-agnostic.

## Runtime concerns for next iteration
- The next runner should verify the new prefix metric magnitude on existing artifacts. It should not saturate near zero for all methods solely because the sample count is high, and it should not dominate the score relative to total variation.
- The evaluator assumes future methods keep the existing artifact contract: `scenario`, `method`, `gamma`, `max_len`, `temperature`, `samples`, `sequence_counts`, `generated_tokens`, and `target_steps`.
- The paper markdown path provided in the prompt was not present in this workspace, so this pass used the existing benchmark files and comments for paper context.
