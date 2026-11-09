#!/bin/bash
# Job entry point: reproduce the reference end to end, score both slices,
# run the hardening checks, then validate the workspace.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

rm -f checkpoints/*.json   # artifacts of the retired histogram contract
bash scripts/reproduce.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash eval/test/hardening/run_hardening.sh
python3 validate.py --post-design
