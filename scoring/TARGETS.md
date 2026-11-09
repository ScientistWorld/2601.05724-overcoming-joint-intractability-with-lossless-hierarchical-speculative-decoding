# Evaluation Targets

Every target is scored by `bash scripts/evaluate_test.sh` (held-out slice,
`scoring/scores_test.json`). `bash scripts/reproduce.sh` exports the reference
and baseline verifier programs and scores the visible slice.

## Primary: single_draft_main
Does a lossless verifier emit more tokens per target step than token-wise
speculative sampling with one draft?
- **Benefit metric**: `block_efficiency` (higher is better; reference 3.3469,
  token-wise 3.1546)
- **Constraint**: exact losslessness (completion total variation <= 1e-9 at every
  context, reported under `constraint`; violation => failed run, 0.0)

## Secondary: single_draft_temperature
Same comparison at temperatures 0.6 and 0.8.
- **Benefit metric**: `block_efficiency` (reference 3.6467, token-wise 3.5041)
- **Constraint**: exact losslessness (as above)

## Secondary: single_draft_length
Same comparison at draft lengths 2–5.
- **Benefit metric**: `block_efficiency` (reference 3.5383, token-wise 3.4109)
- **Constraint**: exact losslessness (as above)

## Primary (headroom): multi_draft
Does the verifier convert several i.i.d. drafts into more kept tokens than
recursive rejection sampling with token-wise checks?
- **Benefit metric**: `block_efficiency` (reference 3.5130, token-wise 3.4233,
  first-draft-only blockwise 3.1881)
- **Constraint**: exact losslessness (as above)
