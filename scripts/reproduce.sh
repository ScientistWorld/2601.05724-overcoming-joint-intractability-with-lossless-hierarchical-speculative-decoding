#!/bin/bash
# Reproduce the paper's results end-to-end on the visible slice.
#
# Exports the reference method (HSD) and the baselines as verifier programs
# under checkpoints/, then scores them with scripts/evaluate.sh
# (writes scoring/scores.json).  Pure Python, CPU only, a few minutes.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

RUN_EVALUATE=0 bash scripts/baseline.sh
RUN_EVALUATE=0 bash scripts/method.sh
bash scripts/evaluate.sh
