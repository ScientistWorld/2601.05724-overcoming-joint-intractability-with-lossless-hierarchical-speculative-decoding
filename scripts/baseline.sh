#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

SCENARIO="${1:-finite_ar_main}"
SAMPLES="${SAMPLES:-50000}"
SEED="${SEED:-101}"

mkdir -p checkpoints
python -m baseline.tokenwise \
  --scenario "${SCENARIO}" \
  --samples "${SAMPLES}" \
  --seed "${SEED}" \
  --output "checkpoints/tokenwise_${SCENARIO}.json"

if [ "${RUN_EVALUATE:-1}" = "1" ]; then
  bash scripts/evaluate.sh
fi
