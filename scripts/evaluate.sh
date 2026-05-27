#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
# Do NOT import from method/ — evaluate.sh must work even when the
# paper's method code is hidden from a scientist agent.
#
# Design it so that anyone following the same output format can run
# this script to evaluate their results.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, e.g.:
#   {"imagenet_classification": {"top1_accuracy": 82.1, "top5_accuracy": 94.8}}
# Experiment and metric names must match the keys in scoring/reference.json.
# This file is read by the system for automated comparison.
#
# Example usage:
#   ./evaluate.sh                                                   # evaluate all checkpoint JSON artifacts
#   ./evaluate.sh checkpoints/hsd_finite_ar_main.json               # evaluate specific artifacts
#
# Called from run.sh after training, or independently.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p scoring

if [ "$#" -gt 0 ]; then
  python -m eval.evaluate --artifacts "$@" --output scoring/scores.json
else
  mapfile -t artifacts < <(find checkpoints -maxdepth 1 -name '*.json' | sort)
  if [ "${#artifacts[@]}" -eq 0 ]; then
    echo "No checkpoint artifacts found. Pass artifact paths or run scripts/method.sh." >&2
    exit 1
  fi
  python -m eval.evaluate --artifacts "${artifacts[@]}" --output scoring/scores.json
fi
