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
efficiency and decoding-speed gains over tokenwise and blockwise verification.

## Key Results

- Average reported improvement is about 6.2% in block efficiency and 6.7% in
  decoding speed across main Qwen2.5 experiments.
- On GSM8K with Qwen2.5-14B, reported block efficiency improves from 5.99
  tokenwise to 6.30 with HSD.
- EAGLE-3 integration reports a 12.4% decoding-speed gain.

---

# Problem Definition

What problem does this paper address? Write this so that someone with no knowledge of the paper can understand what needs to be solved and how success is measured.

Do NOT describe the paper's method or solution here — only the problem itself. Think about if a future scientist agent try to solve this problem from scratch without knowing what approach the paper took. If you leak the solution, you compromise the benchmark.

## Research Question

How can an autoregressive generator reduce the number of serial target-model
decoding steps while still sampling from the same distribution as the target
model? The challenge is that block proposals from a cheaper draft distribution
create joint-probability dependencies that are hard to verify exactly.

## Why It Matters

Autoregressive generation is latency-bound by sequential target-model calls. A
good solution increases generated tokens per target step without changing output
distribution quality.

## Success Criteria

A successful decoder has higher block efficiency than standard tokenwise
speculative decoding while maintaining low distribution distance from direct
target-model sampling.

---

# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

The paper introduces Hierarchical Speculative Decoding (HSD), a lossless
verification method that increases accepted tokens in speculative decoding by
using hierarchical branch resampling rather than independent tokenwise checks.

## Approach

A draft model proposes a block of tokens. The verifier computes joint prefix
ratios under the target and draft distributions, identifies maximum-prefix
capping points, computes capped branch divergences over accessible next-token
branches, and scans backward through the draft to accept the longest validated
prefix. If the full block is accepted, a bonus target token is sampled. If not,
the verifier emits the accepted prefix plus one sample from the capped branch
resampling distribution.

## Main Claims

The primary claim is that HSD improves block efficiency and decoding speed while
remaining lossless, i.e. preserving the target distribution. Secondary claims
include robustness across temperatures and draft lengths, compatibility with
multi-draft decoding, and integration with EAGLE-3.

---

# Evaluation

How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results. Think of this as a general evaluation protocal for other scientists trying to solve the same problem, where they should not see this paper's solution.

## Metrics

- `block_efficiency`: generated tokens per serial target-model step. Higher is
  better.
- `total_variation`: empirical total-variation distance between generated
  sequence counts and the exact target distribution. Lower is better.

## Baselines and Targets

The primary baseline is tokenwise speculative decoding. The paper reports HSD
improvements over tokenwise block efficiency across GSM8K, HumanEval, and
CNN/DailyMail; this workspace uses those reported block-efficiency values as
reference targets while running a finite scaled benchmark.

## Evaluation Protocol

Run `scripts/reproduce.sh` to generate baseline and candidate artifacts, then
run `scripts/evaluate.sh`. Evaluation reads JSON artifacts from `checkpoints/`
and writes `scoring/scores.json`. Candidate methods must use the same finite
target/draft distributions, sequence length, temperature, and draft length for
each named experiment.

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


---

# Reproduction Milestones

**Current: method_runs**

## Stop Justification
- Completed at milestone `method_runs`.
