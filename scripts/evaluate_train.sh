#!/bin/bash
# Visible train-slice evaluator.
#
# Mirrors scripts/evaluate.sh: pass checkpoint JSON artifacts explicitly, or
# omit arguments to evaluate every JSON artifact under checkpoints/.

set -e

cd /home/user

mkdir -p scoring

if [ "$#" -gt 0 ]; then
  python -m eval.train.evaluate --artifacts "$@" --output scoring/scores_train.json
else
  mapfile -t artifacts < <(find checkpoints -maxdepth 1 -name '*.json' | sort)
  if [ "${#artifacts[@]}" -eq 0 ]; then
    echo "No checkpoint artifacts found. Pass artifact paths or run scripts/method.sh." >&2
    exit 1
  fi
  python -m eval.train.evaluate --artifacts "${artifacts[@]}" --output scoring/scores_train.json
fi
