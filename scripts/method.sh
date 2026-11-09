#!/bin/bash
# Run the paper's proposed method: export the HSD verifier program.
#
# The method needs no training.  Its artifact is the self-contained verifier
# program checkpoints/hsd/verifier.py, which the evaluators run on their own
# instances (train: visible models in data/, test: hidden models).
#
# Usage:
#   bash scripts/method.sh            # export, then score on the visible slice
#   RUN_EVALUATE=0 bash scripts/method.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p checkpoints/hsd
cp method/hsd_verifier.py checkpoints/hsd/verifier.py
echo "exported checkpoints/hsd/verifier.py"

if [ "${RUN_EVALUATE:-1}" = "1" ]; then
  bash scripts/evaluate.sh
fi
