#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# Scores every verifier program checkpoints/<method>/verifier.py on the
# visible instances (eval/train, models from data/synthetic_lm.py) and writes
# scoring/scores.json.  Uses only code in eval/ and data/ (never method/).
#
# Usage:
#   bash scripts/evaluate.sh                      # every method under checkpoints/
#   bash scripts/evaluate.sh checkpoints/hsd      # specific method directories

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p scoring
python3 -m eval.train.evaluate "$@" --output scoring/scores.json
