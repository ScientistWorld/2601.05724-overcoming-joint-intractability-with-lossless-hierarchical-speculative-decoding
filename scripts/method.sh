#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

SCENARIO="${1:-finite_ar_main}"
SAMPLES="${SAMPLES:-50000}"
SEED="${SEED:-1}"

mkdir -p checkpoints
python -m method.hsd \
  --scenario "${SCENARIO}" \
  --samples "${SAMPLES}" \
  --seed "${SEED}" \
  --output "checkpoints/hsd_${SCENARIO}.json"

bash scripts/evaluate.sh
