# Problem Definition

## Research Question

How can an autoregressive generator reduce the number of serial target-model
decoding steps while still sampling from the same distribution as the target
model? The challenge is that multiple proposed tokens can have dependencies
that are easy to exploit for speed but hard to validate without biasing the
output distribution.

## Why It Matters

Autoregressive generation is latency-bound by sequential target-model calls. A
good solution increases generated tokens per target step without changing output
distribution quality.

## Success Criteria

A successful decoder produces more tokens per serial target step than a
standard token-by-token speculative baseline while maintaining low distribution
distance from direct target-model sampling.
