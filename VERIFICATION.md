# Verification Report

## Initial Assessment
The workspace started at `core_claim_plus` with a completed scaled finite
autoregressive reproduction of the paper's verifier claim. The claimed
milestone is justified for this gym: the method code implements the paper's
actual verifier computation at finite scale, produces artifacts, and the
evaluation reads those artifacts without importing `method/`.

The reproduced scores show the intended tradeoff across all three configured
settings: higher block efficiency than the tokenwise baseline, with empirical
distribution distance measured against the exact finite target distribution.

## Issues Found
- **Minor:** `briefing/evaluation.md` still referred to the original paper in a
  method-agnostic file. It did not leak the solution algorithm, but it added
  unnecessary source-paper framing for from-scratch agents.
- **Minor:** `scoring/DIRECTION.md` used one phrase that was closer to a
  verifier-design detail than needed for high-level research direction.
- **Cosmetic / pre-existing:** The worktree still has unrelated dirty
  system/workflow files (`README.md`, `manifest.json`, and deleted
  `trajectory/*`). I did not revert them.

## Fixes Applied
- Reworded `briefing/evaluation.md` so it describes the finite benchmark and
  metrics without source-paper framing.
- Reworded `scoring/DIRECTION.md` to keep the allowed research direction at the
  level of speculative-decoding verification and probability accounting.
- Re-ran `bash scripts/evaluate.sh`, `bash scripts/evaluate_train.sh
  checkpoints`, `bash scripts/evaluate_test.sh checkpoints`, and `python
  validate.py --compare`; all passed.

## Final Judgment
The workspace is ready to serve as a research gym. Improve-mode agents can use
the implemented method, baseline, scores, and documentation. From-scratch agents
can use the method-agnostic problem statement, finite data generator, evaluation
code, and scoring instructions without depending on `method/`.

The scoring captures both the benefit metric (`block_efficiency`) and the
fidelity constraints (`total_variation` plus split diagnostics). No scored
metric measures deviation from the paper's published values or wall-clock
speed. The experiment weights emphasize the main finite benchmark while keeping
the draft-length and temperature settings active.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: high
- Key limitation: scaled finite autoregressive benchmark only; the original
  large-model Qwen tables and EAGLE-3 integration are not reproduced.
