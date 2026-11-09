#!/bin/bash
# Export the paper's baseline verifier programs.
#
#   tokenwise  - token-wise speculative sampling (Leviathan et al., 2023);
#                with several drafts, recursive reject sampling with
#                replacement (the paper's multi-draft baseline).
#   blockwise  - block verification (Sun et al., 2024); first draft only.
#
# Usage:
#   bash scripts/baseline.sh             # export both, then score on the visible slice
#   bash scripts/baseline.sh tokenwise   # export one
#   RUN_EVALUATE=0 bash scripts/baseline.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

if [ "$#" -gt 0 ]; then
  names=("$@")
else
  names=(tokenwise blockwise)
fi

for name in "${names[@]}"; do
  src="baseline/${name}_verifier.py"
  if [ ! -f "${src}" ]; then
    echo "unknown baseline: ${name}" >&2
    exit 1
  fi
  mkdir -p "checkpoints/${name}"
  cp "${src}" "checkpoints/${name}/verifier.py"
  echo "exported checkpoints/${name}/verifier.py"
done

if [ "${RUN_EVALUATE:-1}" = "1" ]; then
  bash scripts/evaluate.sh
fi
