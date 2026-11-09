# Research Direction

## Research Problem
Verifier design for lossless speculative decoding: keep as many proposed tokens
per target-model step as possible while sampling exactly from the target model,
using only the probabilities available along the proposals.

## Core Contribution Scope
Improvements should be verification rules: acceptance rules, probability
accounting across positions or across proposals, residual/correction
distributions, and how several proposals are combined. With one proposal per
step there is a known efficiency optimum (the reference attains it), so gains
there can only tie. Several proposals per step leave measurable room.

## Out Of Bounds
- Changing the target/draft models, vocabulary, draft lengths, number of
  drafts or temperatures (they are fixed by the evaluators).
- Lossy shortcuts (accepting more than the distribution allows); they are
  scored as failed.
- Using information outside the request: reading evaluator files, or
  accumulating rows across requests to reconstruct branches a real verifier
  cannot see.
- Optimising wall-clock speed; only tokens per target step are scored.
