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
