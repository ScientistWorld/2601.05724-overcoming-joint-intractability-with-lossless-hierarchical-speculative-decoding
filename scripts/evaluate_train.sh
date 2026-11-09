#!/bin/bash
# Visible train-slice evaluator.
#
# Scores every verifier program checkpoints/<method>/verifier.py (or the
# method directories given as arguments) on the visible instances and writes
# scoring/scores_train.json.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p scoring
python3 -m eval.train.evaluate "$@" --output scoring/scores_train.json
