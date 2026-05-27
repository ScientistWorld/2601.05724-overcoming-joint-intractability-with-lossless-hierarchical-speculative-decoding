# Evaluation Targets

## Primary: finite_ar_main
Tests whether a decoder improves generated tokens per serial target step on the
main finite autoregressive benchmark while preserving the target distribution.
- **Benefit metric**: `block_efficiency` (higher is better)
- **Constraint metric**: `total_variation` (lower is better; empirical values
  are interpreted with Monte Carlo sampling noise)
- **Run**: `bash scripts/reproduce.sh && bash scripts/evaluate.sh`

## Primary: finite_ar_gamma3
Tests whether the same efficiency and fidelity goals hold when the draft block
is shorter.
- **Benefit metric**: `block_efficiency` (higher is better)
- **Constraint metric**: `total_variation` (lower is better; empirical values
  are interpreted with Monte Carlo sampling noise)
- **Run**: `bash scripts/reproduce.sh && bash scripts/evaluate.sh`

## Primary: finite_ar_temp08
Tests whether the same efficiency and fidelity goals hold under a lower sampling
temperature.
- **Benefit metric**: `block_efficiency` (higher is better)
- **Constraint metric**: `total_variation` (lower is better; empirical values
  are interpreted with Monte Carlo sampling noise)
- **Run**: `bash scripts/reproduce.sh && bash scripts/evaluate.sh`
