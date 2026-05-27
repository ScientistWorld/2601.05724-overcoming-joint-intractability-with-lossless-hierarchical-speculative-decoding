# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-05-27 06:42] - core_claim_plus
- Implemented and packaged finite autoregressive HSD reproduction across three settings.
- Verified HSD improves block efficiency over tokenwise speculative decoding in all three settings.
- Confirmed distribution fidelity is measured with empirical total variation against the exact target distribution; lightweight exact enumeration found HSD length-4 output TV at numerical precision.

### [2026-05-27 06:44] - core_claim_plus
- Continuing-agent audit confirmed the implementation is a real scaled HSD verifier rather than a surrogate.
- Fixed standalone baseline script evaluation behavior and clarified finite-reference documentation.
- Re-ran `scripts/evaluate.sh` and `python validate.py --compare`; both passed.
- Milestone is restored for the scaled finite-AR verifier benchmark, not for the original Qwen large-model tables or EAGLE-3 integration.

## Stop Justification
- Completed at milestone `core_claim_plus`.
