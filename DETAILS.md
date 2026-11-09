# Overview

- **Paper ID:** 2601.05724
- **Title:** Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding
- **Domain:** Language model inference / speculative decoding
- **TL;DR:** HSD is a lossless speculative decoding verifier that accepts longer draft prefixes by redistributing branch probability mass hierarchically.

## Short Summary

The paper targets the verification bottleneck in speculative decoding. It
proposes capped branch resampling, which lets the verifier perform a backward
block-level acceptance scan and still recover the target distribution with a
single resampling step. The reported experiments show consistent block
efficiency and decoding-speed gains over tokenwise and blockwise verification,
and gains when combined with multi-draft recursive rejection sampling.

## Key Results (paper)

- Average reported improvement is about 6.2% in block efficiency and 6.7% in
  decoding speed across main Qwen2.5 experiments.
- On GSM8K with Qwen2.5-14B, reported block efficiency improves from 5.99
  tokenwise to 6.30 with HSD.
- Multi-draft (11 drafts): +5.9% block efficiency over tokenwise multi-draft.
- EAGLE-3 integration reports a 12.4% decoding-speed gain.

## This gym

Synthetic target/draft pairs with exact scoring. Methods are verifier
programs run on hidden model pairs (see `briefing/evaluation.md`). Four
experiments: single-draft main, temperature, draft length, and multi-draft.

---

# Problem Definition

## Research Question

How can an autoregressive generator emit more tokens per serial call to a large
target model while sampling **exactly** from the target model's distribution?
A cheap draft model proposes one or more blocks of tokens. A single parallel
target pass then gives the target's next-token probabilities along every
proposed block. A verifier must decide how much of a proposal to keep and which
single new token to add. The joint probability of an alternative continuation
is not available: only the rows along the proposed blocks are. The difficulty
is to keep long proposals as often as possible without biasing the output
distribution.

## Why It Matters

Autoregressive generation is latency-bound by sequential target-model calls.
Every extra token kept per call is a direct speed-up, but only if output
quality is unchanged, i.e. the decoder is lossless.

## Setting in This Benchmark

Target/draft model pairs are synthetic next-token tables over small
vocabularies (3–6 tokens), so the output distribution and the expected number
of tokens per step can be computed exactly. Settings cover:

* one proposal per step (main setting, lower temperatures, several proposal
  lengths);
* several independent proposals per step (multi-draft).

## Success Criteria

A successful verifier is exactly lossless in every setting and emits more tokens
per target step than standard token-by-token speculative sampling. A better
verifier emits more tokens per step on held-out model pairs.

---

# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

Hierarchical Speculative Decoding (HSD) is a lossless verifier that uses
hierarchical branch resampling instead of independent token-wise checks, so it
can accept longer draft prefixes.

## Approach (single draft, Algorithm 2)

Along the draft the verifier computes joint prefix ratios r(X_{1:i}) =
p(X_{1:i})/q(X_{1:i}). It caps them at the maximum-prefix-ratio point
(Definitions 4–5), computes capped branch divergences over each accessible
next-token branch (Definition 6), and scans backward from the end of the draft,
accepting prefix X_{1:t} with probability h_t (Equation 19). If the whole draft
is accepted, a bonus token is sampled from the target. Otherwise one token is
drawn from the capped branch resampling distribution (Equation 20).

## Multi-draft (Algorithm 3)

Drafts are visited in order. A draft is verified only if it extends the
currently accepted prefix, starting from that node with the previous capped
residual installed as the node's target distribution (recursive rejection
sampling with replacement). The final residual is sampled when the drafts run
out.

## Implementation in this workspace

`method/hsd_verifier.py` implements Algorithms 2 and 3 and reports their exact
outcome distribution (protocol in `briefing/evaluation.md`).
`scripts/method.sh` exports it to `checkpoints/hsd/verifier.py`.

## Main Claims

HSD improves block efficiency and decoding speed over token-wise and blockwise
verification while remaining lossless. It is robust across temperatures and
draft lengths, and it also improves multi-draft decoding and EAGLE-3.

## Notes from this reproduction

* Single-draft HSD (Algorithm 2) is algebraically identical to block
  verification (Sun et al., 2024). The capped scale r(X_{1:t})/max(1, max_{i<=t}
  r(X_{1:i})) equals block verification's running term p_t, and the
  denominators of h_t agree. The two methods therefore have identical expected
  efficiency. The paper's reported HSD-over-Blockwise margins are not reproduced
  (exact tie).
* HSD over token-wise verification reproduces in every setting (+3.7% to +6.1%
  single draft, +2.6% multi-draft on the held-out slice).

---

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

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 1 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-27 00:00] - method_runs
- Read the paper and identified HSD's capped branch resampling verifier as the core algorithm.
- Implemented HSD and tokenwise speculative decoding on a finite autoregressive benchmark.
- Ran a local smoke test showing both methods execute end-to-end and HSD improves block efficiency on the main finite setting.

### [2026-05-27 06:22] - core_claim
- CPU reproduction job `291c5539-26f` completed successfully.
- HSD improved block efficiency over tokenwise on the main finite setting: 3.2886 vs 3.1425 tokens/step.
- HSD also improved the two configured robustness settings while keeping total-variation distance comparable to tokenwise.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-27 06:35] - method_runs
- Continuing-agent audit found the implementation is a real HSD verifier rather than a surrogate.
- Standalone evaluation and validation work, but a smoke run overwrote local artifacts with low-sample results; resubmitting the full CPU reproduction before restoring `core_claim`.

### [2026-05-27 06:40] - core_claim_plus
- CPU rerun job `5f35dfcd-72a` completed successfully with 80,000 samples per method/setting.
- The rerun reproduced HSD's block-efficiency advantage on the main finite benchmark and on the shorter-draft and lower-temperature settings.
- `python validate.py --compare` passed after the rerun.

### [2026-05-27 06:26] - method_runs
- Implemented the paper's HSD verifier with capped prefix ratios, capped branch divergences, backward verification, and single capped resampling.
- Implemented tokenwise speculative decoding baseline and artifact-based evaluation on finite autoregressive target/draft distributions.

### [2026-05-27 06:26] - core_claim
- Ran an 80,000-sample CPU reproduction job on `finite_ar_main`.
- HSD improved block efficiency over tokenwise speculative decoding while maintaining comparable total-variation distance to the exact target distribution.

### [2026-05-27 06:26] - core_claim_plus
- Extended the same comparison to `finite_ar_gamma3` and `finite_ar_temp08`.
- Fixed the evaluator to enforce fixed benchmark scenario settings from `data/finite_ar.py`.
- Fixed an HSD acceptance edge case for zero capped reverse branch divergence.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-27 06:42] - core_claim_plus
- Implemented and packaged finite autoregressive HSD reproduction across three settings.
- Verified HSD improves block efficiency over tokenwise speculative decoding in all three settings.
- Confirmed distribution fidelity is measured with empirical total variation against the exact target distribution; lightweight exact enumeration found HSD length-4 output TV at numerical precision.

### [2026-05-27 06:44] - core_claim_plus
- Continuing-agent audit confirmed the implementation is a real scaled HSD verifier rather than a surrogate.
- Fixed standalone baseline script evaluation behavior and clarified finite-reference documentation.
- Re-ran `scripts/evaluate.sh` and `python validate.py --compare`; both passed.
- Milestone is restored for the scaled finite-AR verifier benchmark, not for the original Qwen large-model tables or EAGLE-3 integration.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-27 06:42] - core_claim_plus
- Implemented and packaged finite autoregressive HSD reproduction across three settings.
- Verified HSD improves block efficiency over tokenwise speculative decoding in all three settings.
- Confirmed distribution fidelity is measured with empirical total variation against the exact target distribution; lightweight exact enumeration found HSD length-4 output TV at numerical precision.

### [2026-05-27 06:44] - core_claim_plus
- Continuing-agent audit confirmed the implementation is a real scaled HSD verifier rather than a surrogate.
- Fixed standalone baseline script evaluation behavior and clarified finite-reference documentation.
- Re-ran `scripts/evaluate.sh` and `python validate.py --compare`; both passed.
- Milestone is restored for the scaled finite-AR verifier benchmark, not for the original Qwen large-model tables or EAGLE-3 integration.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-27 06:42] - core_claim_plus
- Implemented and packaged finite autoregressive HSD reproduction across three settings.
- Verified HSD improves block efficiency over tokenwise speculative decoding in all three settings.
- Confirmed distribution fidelity is measured with empirical total variation against the exact target distribution; lightweight exact enumeration found HSD length-4 output TV at numerical precision.

### [2026-05-27 06:44] - core_claim_plus
- Continuing-agent audit confirmed the implementation is a real scaled HSD verifier rather than a surrogate.
- Fixed standalone baseline script evaluation behavior and clarified finite-reference documentation.
- Re-ran `scripts/evaluate.sh` and `python validate.py --compare`; both passed.
- Milestone is restored for the scaled finite-AR verifier benchmark, not for the original Qwen large-model tables or EAGLE-3 integration.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-27 06:42] - core_claim_plus
- Implemented and packaged finite autoregressive HSD reproduction across three settings.
- Verified HSD improves block efficiency over tokenwise speculative decoding in all three settings.
- Confirmed distribution fidelity is measured with empirical total variation against the exact target distribution; lightweight exact enumeration found HSD length-4 output TV at numerical precision.

### [2026-05-27 06:44] - core_claim_plus
- Continuing-agent audit confirmed the implementation is a real scaled HSD verifier rather than a surrogate.
- Fixed standalone baseline script evaluation behavior and clarified finite-reference documentation.
- Re-ran `scripts/evaluate.sh` and `python validate.py --compare`; both passed.
- Milestone is restored for the scaled finite-AR verifier benchmark, not for the original Qwen large-model tables or EAGLE-3 integration.

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-27 06:42] - core_claim_plus
- Implemented and packaged finite autoregressive HSD reproduction across three settings.
- Verified HSD improves block efficiency over tokenwise speculative decoding in all three settings.
- Confirmed distribution fidelity is measured with empirical total variation against the exact target distribution; lightweight exact enumeration found HSD length-4 output TV at numerical precision.

### [2026-05-27 06:44] - core_claim_plus
- Continuing-agent audit confirmed the implementation is a real scaled HSD verifier rather than a surrogate.
- Fixed standalone baseline script evaluation behavior and clarified finite-reference documentation.
- Re-ran `scripts/evaluate.sh` and `python validate.py --compare`; both passed.
- Milestone is restored for the scaled finite-AR verifier benchmark, not for the original Qwen large-model tables or EAGLE-3 integration.

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

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

</details>


---

# Reproduction Milestones

**Current: core_claim_plus**

## Stop Justification
- Completed at milestone `core_claim_plus`.
