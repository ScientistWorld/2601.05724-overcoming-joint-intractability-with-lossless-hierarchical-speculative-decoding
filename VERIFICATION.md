# Verification Report (iteration 11, hardened gym)

## Assessment
- `method/hsd_verifier.py` implements the paper's Algorithm 2 and Algorithm 3.
  Its acceptance probabilities equal the previous sampling implementation
  to 9.2e-15 (16,436 draft positions).
- The evaluators (`eval/train`, `eval/test`) never import `method/`. They run
  each `checkpoints/<name>/verifier.py` as a subprocess and recompute every
  score from its outcome distributions. `eval/test` imports nothing from
  `data/` or `eval/train/`.
- Scores are exact (full draft enumeration); losslessness is a hard gate.
- Job `28904502-86a`: reproduce, train/test scoring, hardening runs, and
  `python3 validate.py --post-design` all passed.

## Known limitations
- Synthetic small-vocabulary model pairs instead of Qwen2.5 LLMs; no decoding
  speed or EAGLE-3.
- Single-draft experiments are at a proved optimum (reference = block
  verification); headroom is in `multi_draft`.
- See `HARDENING.md` §8 for shortcuts left to the grader.

- Milestone: secondary_claims
- Ready for gym use: yes
