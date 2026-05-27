#!/bin/bash
# Visible train-slice evaluator.
#
# Mirrors scripts/evaluate.sh: pass checkpoint JSON artifacts explicitly, or
# omit arguments to evaluate every JSON artifact under checkpoints/.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p scoring

artifacts=()
if [ "$#" -gt 0 ]; then
  for input in "$@"; do
    if [ -d "$input" ]; then
      while IFS= read -r artifact; do
        artifacts+=("$artifact")
      done < <(find "$input" -maxdepth 1 -type f -name '*.json' | sort)
    else
      artifacts+=("$input")
    fi
  done
else
  mapfile -t artifacts < <(find checkpoints -maxdepth 1 -type f -name '*.json' | sort)
fi

if [ "${#artifacts[@]}" -eq 0 ]; then
  echo "No checkpoint artifacts found. Pass artifact paths, a checkpoint directory, or run scripts/method.sh." >&2
  exit 1
fi

python -m eval.train.evaluate --artifacts "${artifacts[@]}" --output scoring/scores_train.json
