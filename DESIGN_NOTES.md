# Train/Test Evaluation Design Notes (hardened, iteration 11)

## Contract
- Methods are verifier programs `checkpoints/<name>/verifier.py` (JSON-lines
  protocol, `briefing/evaluation.md`). Evaluators discover every such directory;
  no method name is hardcoded.
- Each program runs in its own subprocess (`python3 -I`, empty working
  directory). Problems are streamed one at a time in a random global order
  across all contexts and experiments; 2% are re-asked later as a determinism
  probe.
- Scores are recomputed by the evaluator from the outcome distributions. For
  each context every draft tuple is enumerated, so `block_efficiency` and the
  losslessness check are exact. A protocol violation, invalid probabilities,
  nondeterminism, a crash, a timeout or completion TV > 1e-9 makes the method
  `failed` (0.0 in every experiment).

## Split
- `eval/train/evaluate.py`: visible instances from `data/synthetic_lm.py`
  (salt/seed visible, 3 contexts per single-draft setting, 1 per multi-draft
  setting).
- `eval/test/evaluate.py`: self-contained; private copy of the model family,
  hidden salt/seed, 4 / 2 contexts per setting. It imports nothing from `data/`,
  `method/` or `eval/train/`, so editing visible files cannot move test scores.
- Both slices compute a fixed token-wise baseline in-process (`fixed_baseline`).

## Hardening material
`eval/test/hardening/` (hidden with the test slice) holds the better/weaker
variants, the shortcut programs, `run_hardening.sh`, `rank.py` and the results
reported in `HARDENING.md`.
