# Scientist Constraints

## Model And Data
The target and draft models are fixed by the evaluators: visible pairs come from
`data/synthetic_lm.py`, held-out pairs are generated inside the test evaluator.
A method only ever sees what a verifier sees in speculative decoding: the draft
blocks and the target/draft next-token rows along them. Editing
`data/synthetic_lm.py` changes only the visible slice, never the held-out score.

## Method Contract
A method is a self-contained, standard-library Python program
`checkpoints/<name>/verifier.py` that speaks the JSON-lines protocol in
`briefing/evaluation.md`. It must:
- report the exact probability of each outcome `[draft_index, tau, token, prob]`
  (finite, non-negative, summing to 1);
- be a deterministic function of the request (repeated requests are checked);
- not read evaluator files or any information beyond the request, and not keep
  information across requests to reconstruct unseen branches;
- finish a slice within the time budget (default 5400 s).

## Fidelity Constraint
Losslessness is checked exactly at every held-out context: emitted tokens plus
target continuation must reproduce the target distribution (total variation
<= 1e-9). Any violation scores the method as failed (block efficiency 0.0 in all
experiments). A faster but slightly biased verifier is a failed verifier.

## Fixed Settings
Vocabulary sizes, draft lengths, number of drafts, temperatures, contexts and
the token-wise reference value (`fixed_baseline`) are set by the evaluators and
cannot be changed by a method.

## Compliance Checklist
- [x] `checkpoints/hsd/verifier.py` exported by `scripts/method.sh`
- [x] Baselines exported by `scripts/baseline.sh`
- [x] Lossless in every experiment (max completion TV about 1e-15)
- [x] Scores produced only by `scripts/evaluate*.sh`
