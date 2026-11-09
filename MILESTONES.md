# Reproduction Milestones

**Current: secondary_claims**

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

### [2026-05-27 07:09] - core_claim_plus
- Independent explorer audit reconfirmed `method/hsd.py` implements capped-ratio HSD and `eval/` does not import `method/`.
- Fixed default `scores.json` to include all diagnostic metric keys declared in `reference.json`.
- Re-ran `bash scripts/evaluate.sh` and `python validate.py --compare`; both passed.

### [2026-05-27 07:12] - core_claim_plus
- Final verification confirmed the claimed milestone is justified for the scaled finite autoregressive gym.
- Removed method leakage from method-agnostic briefing files and added explicit finite-scaling notes to `scoring/reference.json`.
- Re-ran `bash scripts/evaluate.sh` and `python validate.py --compare`; both passed.

### [2026-05-27 07:14] - core_claim_plus
- Sub-agent audits reconfirmed `method/hsd.py` implements the paper's capped-branch HSD verifier rather than a surrogate.
- Reconfirmed `scripts/evaluate.sh` reads artifacts through `eval/` without importing `method/`.
- Marked the satisfied scientist constraints checklist and re-ran `bash scripts/evaluate.sh` plus `python validate.py --compare`; both passed.

### [2026-05-27 07:18] - core_claim_plus
- Final quality check confirmed problem/solution separation, artifact-based evaluation, train/test split wiring, and shared-path hygiene.
- Tightened method-agnostic documentation in `briefing/evaluation.md` and `scoring/DIRECTION.md`.
- Re-ran `bash scripts/evaluate.sh`, train/test evaluators, and `python validate.py --compare`; all passed.

### [2026-11-09] - secondary_claims (iteration 11, hardening)
- Replaced the histogram/counter contract with verifier programs scored exactly on hidden target/draft pairs; losslessness is an exact hard gate.
- Found that HSD (Algorithm 2) is algebraically identical to block verification (Sun et al. 2024): it ties blockwise exactly and sits at the single-draft optimum.
- Reproduced HSD > tokenwise on the held-out slice: +6.10% (main), +4.07% (temperature), +3.74% (draft length), +2.62% (multi-draft, Algorithm 3, 18/18 settings).
- Added multi_draft as the headroom experiment: an optimal-coupling tree verifier beats the reference by 5.3% (18/18 settings); all ported shortcuts score as failed.

## Stop Justification
- Completed at milestone `core_claim_plus`.
