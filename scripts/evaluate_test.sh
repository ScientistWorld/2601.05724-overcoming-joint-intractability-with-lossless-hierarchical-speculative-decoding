#!/bin/bash
# Held-out test-slice evaluator.
#
# With no arguments, scores every verifier program checkpoints/<method>/verifier.py
# on the hidden instances generated inside eval/test/ and writes
# scoring/scores_test.json.  Arguments, if given, are method directories (or
# roots of method directories) to score instead.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p scoring
python3 -m eval.test.evaluate "$@" --output "${SCORES_TEST_OUT:-scoring/scores_test.json}"
