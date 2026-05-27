# Verification Report

## Initial Assessment
The workspace started at `core_claim_plus` with a working scaled finite
autoregressive reproduction of the paper's speculative-decoding verifier claim.
The claimed milestone is justified: the method implementation executes the
paper's verifier computation on a finite exact-probability benchmark, HSD
improves block efficiency over tokenwise speculative decoding across three
settings, and fidelity is measured against the exact target distribution.

## Issues Found
- **Minor:** `briefing/evaluation.md` named the paper's method in a
  method-agnostic file. This could leak solution identity to from-scratch
  agents.
- **Minor:** `briefing/problem.md` still contained template instructions and
  could describe the challenge more cleanly without solution-specific phrasing.
- **Minor:** `scoring/reference.json` used deliberate finite-benchmark
  reference values instead of the paper's large-model table values, but the
  scaling rationale was only in descriptions rather than explicit `notes`
  fields.
- **Cosmetic / pre-existing:** The git worktree still contains unrelated dirty
  files from earlier workflow state (`README.md`, `manifest.json`, deleted
  `trajectory/*`). I did not revert or include them.

## Fixes Applied
- Rewrote `briefing/problem.md` and `briefing/evaluation.md` to keep the
  problem and evaluation method-agnostic.
- Added explicit per-experiment `notes` in `scoring/reference.json` documenting
  that reference values are from the validated finite scaled benchmark, not the
  original Qwen table values.
- Re-ran `bash scripts/evaluate.sh` and `python validate.py --compare`; both
  pass.

## Final Judgment
The workspace is ready to serve as a research gym. Improve-mode agents can build
on the implemented method and compare against the existing artifacts.
From-scratch agents can use the method-agnostic problem, data, and evaluation
pipeline without importing or relying on `method/`. The main limitation is
scope: this is a scaled finite-distribution verifier benchmark, not a
reproduction of the paper's full large-model Qwen or EAGLE-3 experiments.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: high
- Key limitation: scaled finite autoregressive benchmark only; original
  large-model tables are not reproduced.
