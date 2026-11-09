# Experiment Overview

All experiments score one quantity, `block_efficiency`: the exact expected
number of tokens emitted per target-model step on held-out target/draft model
pairs. Each also carries one pass/fail constraint, exact losslessness: the
decoder's output must follow the target distribution. A method that violates
the constraint in any experiment, or breaks the protocol, is scored as failed
(0.0 everywhere). Scores are exact (every draft is enumerated), so there is no
seed noise. Rerunning in a different problem order changes them by < 2e-14.

## single_draft_main (weight 0.35)
Establishes the central claim: with one proposed block per step, a lossless
verifier should keep more tokens than standard token-by-token speculative
sampling. The block length is 4 over a 6-token vocabulary at temperature 1, on
three content regimes (low, medium and high target entropy) times two
draft/target pairings. For one proposal per step, lossless verification has a
known optimum, and the reference attains it, so this experiment rewards
matching the best possible single-draft verifier and penalises anything worse.

## single_draft_temperature (weight 0.10)
Checks that the efficiency advantage survives sharper sampling (temperatures
0.6 and 0.8 applied to both models). Same single-draft optimum as above.

## single_draft_length (weight 0.15)
Checks that the advantage holds across proposal lengths 2, 3, 4 and 5 (vocabulary
5). Same single-draft optimum as above.

## multi_draft (weight 0.40)
Several independent proposals per step: 2 blocks of length 4 over 3 tokens,
2 blocks of length 3 over 4 tokens, and 3 blocks of length 3 over 3 tokens,
across three regimes and two pairings. This tests whether a verifier turns
extra proposals into extra kept tokens without bias. No optimum is known here,
and the reference is measurably below what is achievable, so this experiment
is where better verifiers score higher.
