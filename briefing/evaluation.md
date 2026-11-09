# Evaluation

## What a method submits

A method is a **verifier program**: a self-contained Python file
`checkpoints/<method_name>/verifier.py` (standard library only). The
evaluators find every such directory, so a new method needs no registration.
`scripts/method.sh` and `scripts/baseline.sh` show how the shipped programs are
exported. A method needs no training; the program is run by the evaluator on
its own instances.

The program is started as `python3 -I verifier.py` in an empty working
directory and talks over stdin/stdout, one JSON object per line. It must
answer each request before it receives the next one.

Request (one per target-model step):

```json
{"id": 17, "vocab_size": 6, "gamma": 4, "num_drafts": 1,
 "drafts":       [[3, 1, 0, 5]],
 "target_probs": [[[...V floats...], ... gamma+1 rows]],
 "draft_probs":  [[[...V floats...], ... gamma rows]]}
```

* `drafts[k]` is the k-th draft block, `gamma` tokens sampled from the draft
  model (all K blocks are independent draws from the same context).
* `target_probs[k][t]` is the target model's next-token distribution after the
  context and the first `t` tokens of draft k (t = 0..gamma).
* `draft_probs[k][t]` is the draft model's next-token distribution at the same
  positions (t = 0..gamma-1).

Nothing else about the models is available. These are exactly the quantities
one draft pass and one parallel target pass produce in speculative decoding.

Response:

```json
{"id": 17, "outcomes": [[k, tau, y, prob], ...]}
```

Each outcome means that this target step emits the tokens
`drafts[k][:tau] + [y]` (0 <= tau <= gamma). `prob` is the probability that
your decoder produces this outcome for this request; the probabilities must be
finite, non-negative and sum to 1. Report the exact distribution your
randomised decoder would sample from; the evaluator does not sample. Answers
must be a deterministic function of the request. Some requests are repeated
and must get the same answer.

## Metric

`block_efficiency` (higher is better): the expected number of tokens emitted
per target-model step. For every context of an experiment the evaluator
enumerates **every** possible draft tuple, weighted by its draft-model
probability, and averages the emitted length of your outcomes. The result is
exact, with no sampling noise. The experiment value is the mean over its
contexts.

## Constraint (pass/fail)

The output must follow the target model's distribution exactly: for every
context, the emitted tokens followed by ordinary sampling from the target
model must reproduce the target distribution over `gamma + 1` tokens (total
variation <= 1e-9, checked exactly). A program that fails this anywhere is
scored as **failed**: `block_efficiency = 0.0` in every experiment. Any valid
decoder scores at least 1.0. The same failure applies to malformed or
out-of-range outcomes, non-finite or unnormalised probabilities, answers that
change between repeated requests, crashes, and exceeding the time budget
(default 5400 s per program for a whole slice; the shipped programs need
1–2 minutes).

## Slices

* `scripts/evaluate.sh` / `scripts/evaluate_train.sh`: visible instances
  (model families and seeds in `data/synthetic_lm.py`), writing
  `scoring/scores.json` / `scoring/scores_train.json`.
* `scripts/evaluate_test.sh`: held-out instances of the same families with
  hidden seeds and more contexts, writing `scoring/scores_test.json`. This is
  the score that ranks methods.

Score files also report, per experiment, the evaluator's own token-by-token
speculative decoding value (`fixed_baseline`), each method's worst completion
total variation (`constraint`) and per-setting breakdowns (`breakdown`).

## Baselines and reference

`tokenwise` (standard token-by-token speculative sampling; with several drafts,
recursive rejection sampling over the drafts) and `blockwise` (block-level
verification of the first draft) are shipped baselines. The reference method's
held-out values are in `scoring/reference.json`.
