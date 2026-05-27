#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

SCENARIO="${1:-finite_ar_main}"
SAMPLES="${SAMPLES:-50000}"
SEED="${SEED:-1}"

mkdir -p checkpoints
python -m method.hsd \
  --scenario "${SCENARIO}" \
  --samples "${SAMPLES}" \
  --seed "${SEED}" \
  --output "checkpoints/hsd_${SCENARIO}.json"

if [ "${RUN_EVALUATE:-1}" = "1" ]; then
  bash scripts/evaluate.sh
fi
