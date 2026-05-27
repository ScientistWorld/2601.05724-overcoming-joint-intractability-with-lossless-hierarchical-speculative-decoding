# Reproduction Milestones

**Current: core_claim**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-05-27 00:00] - method_runs
- Read the paper and identified HSD's capped branch resampling verifier as the core algorithm.
- Implemented HSD and tokenwise speculative decoding on a finite autoregressive benchmark.
- Ran a local smoke test showing both methods execute end-to-end and HSD improves block efficiency on the main finite setting.

### [2026-05-27 06:22] - core_claim
- CPU reproduction job `291c5539-26f` completed successfully.
- HSD improved block efficiency over tokenwise on the main finite setting: 3.2886 vs 3.1425 tokens/step.
- HSD also improved the two configured robustness settings while keeping total-variation distance comparable to tokenwise.

## Stop Justification
- Completed verifier-focused scaled reproduction at `core_claim`.
