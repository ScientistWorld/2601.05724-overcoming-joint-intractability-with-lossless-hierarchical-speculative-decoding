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
