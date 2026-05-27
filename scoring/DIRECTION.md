# Research Direction

## Research Problem
This benchmark studies verifier design for lossless speculative decoding. The
research question is how to accept more draft tokens per target-model step while
preserving the target sampling distribution.

## Core Contribution Scope
Improvements should operate in the speculative-decoding verification and
resampling family. A valid improvement may change acceptance rules, probability
accounting, proposal verification, or correction distributions, provided the
method still uses draft proposals to accelerate target-distribution sampling.

## Out Of Bounds
- Replacing the problem with ordinary direct target sampling.
- Changing the finite target/draft distributions or benchmark settings to make
  the task easier.
- Optimizing only speed or block efficiency while ignoring distribution
  fidelity.
- Using a different generative model or external data to sidestep verifier
  design.
