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
