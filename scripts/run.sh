#!/bin/bash
# Job entry point: reproduce the reference end to end (export the HSD verifier
# and the baselines, score the visible slice and, when present, the held-out
# slice), then validate the workspace.
#   RUN_HARDENING=1 bash scripts/run.sh   also reruns the hardening checks
#                                         (variants and shortcuts, eval/test/hardening)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

bash scripts/reproduce.sh
bash scripts/evaluate_train.sh
if [ -f eval/test/evaluate.py ]; then
  bash scripts/evaluate_test.sh
  if [ "${RUN_HARDENING:-0}" = "1" ]; then
    bash eval/test/hardening/run_hardening.sh
  fi
fi
python3 validate.py --post-design
