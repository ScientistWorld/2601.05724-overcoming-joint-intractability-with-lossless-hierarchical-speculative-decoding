# Scientist Constraints

## Model And Data
Use the finite autoregressive target and draft distributions defined in
`data/finite_ar.py`. Do not replace them with a larger model, extra data, or a
different vocabulary when reporting the benchmark experiments, because the gym
is designed to isolate verifier behavior rather than model quality.

## Evaluation Protocol
Use the exact experiment names, sequence lengths, temperatures, and draft
lengths in `data/finite_ar.py`. Candidate methods must write artifacts with the
same fields consumed by `eval/evaluate.py`.

## Fidelity Constraint
Distribution fidelity must be measured against the exact target distribution for
the same finite autoregressive process. Do not evaluate only efficiency; a
method that samples from the wrong distribution fails the core constraint even
if block efficiency is higher.

## Compliance Checklist
- [x] Uses the provided finite target and draft distributions unchanged
- [x] Uses the configured `finite_ar_main`, `finite_ar_gamma3`, and `finite_ar_temp08` settings unchanged
- [x] Writes method artifacts for `scripts/evaluate.sh`
- [x] Reports both `block_efficiency` and `total_variation`
