#!/bin/bash
# Held-out test-slice evaluator.
#
# Mirrors scripts/evaluate.sh while matching the experiment/method entries
# recorded by scoring/scores_train.json when that file exists.

set -e

cd /home/user

mkdir -p scoring

if [ "$#" -gt 0 ]; then
  python -m eval.test.evaluate \
    --artifacts "$@" \
    --train-scores scoring/scores_train.json \
    --output scoring/scores_test.json
else
  mapfile -t artifacts < <(find checkpoints -maxdepth 1 -name '*.json' | sort)
  if [ "${#artifacts[@]}" -eq 0 ]; then
    echo "No checkpoint artifacts found. Pass artifact paths or run scripts/method.sh." >&2
    exit 1
  fi
  python -m eval.test.evaluate \
    --artifacts "${artifacts[@]}" \
    --train-scores scoring/scores_train.json \
    --output scoring/scores_test.json
fi
