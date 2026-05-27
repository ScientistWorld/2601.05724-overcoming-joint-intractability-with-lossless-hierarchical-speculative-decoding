# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-05-27 06:35] - method_runs
- Continuing-agent audit found the implementation is a real HSD verifier rather than a surrogate.
- Standalone evaluation and validation work, but a smoke run overwrote local artifacts with low-sample results; resubmitting the full CPU reproduction before restoring `core_claim`.

### [2026-05-27 06:26] - method_runs
- Implemented the paper's HSD verifier with capped prefix ratios, capped branch divergences, backward verification, and single capped resampling.
- Implemented tokenwise speculative decoding baseline and artifact-based evaluation on finite autoregressive target/draft distributions.

### [2026-05-27 06:26] - core_claim
- Ran an 80,000-sample CPU reproduction job on `finite_ar_main`.
- HSD improved block efficiency over tokenwise speculative decoding while maintaining comparable total-variation distance to the exact target distribution.

### [2026-05-27 06:26] - core_claim_plus
- Extended the same comparison to `finite_ar_gamma3` and `finite_ar_temp08`.
- Fixed the evaluator to enforce fixed benchmark scenario settings from `data/finite_ar.py`.
- Fixed an HSD acceptance edge case for zero capped reverse branch divergence.

## Stop Justification
- Completed at milestone `core_claim_plus`. Higher milestones would require reproducing the original large-model Qwen/LLaMA and EAGLE-3 experiments, which exceed the available 2-GPU/15-GPU-hour budget.
