# Experiment Overview

## finite_ar_main
This experiment measures the central tradeoff in speculative decoding on the
default finite autoregressive benchmark: how many tokens are produced per serial
target step, and whether the resulting samples still match the target
distribution.

The fidelity metric is empirical total variation from finite samples.
`scoring/reference.json` uses `total_variation=0.0` as the exact
target-distribution ideal, not as an observed finite-sample value. Exact
enumeration of this finite setup can be used as a diagnostic when distinguishing
sampling noise from a systematic verifier error.

## finite_ar_gamma3
This experiment tests the same tradeoff when proposals are shorter. It checks
whether the efficiency effect remains present outside the default block length.

## finite_ar_temp08
This experiment tests the same tradeoff when the sampling distribution is made
sharper by lowering temperature. It checks whether efficiency gains remain
compatible with distribution fidelity under a different sampling regime.
