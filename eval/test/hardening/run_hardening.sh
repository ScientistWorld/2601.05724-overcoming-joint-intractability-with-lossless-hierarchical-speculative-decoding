#!/bin/bash
# Hardening checks (hidden from agents): run the better/weaker variants and
# the shortcut programs through the hidden test evaluator, show that editing
# the visible model file cannot move test scores, rerun the reference with a
# different problem order, and rank everything.  Requires checkpoints/ to hold
# the reference exports (scripts/reproduce.sh).

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}/../../.."
OUT=eval/test/hardening/results
mkdir -p "${OUT}"

# Reference rerun with a different streaming order (exactness / determinism).
EVAL_STREAM_SEED=12345 SCORES_TEST_OUT="${OUT}/hsd_rerun.json" bash scripts/evaluate_test.sh checkpoints/hsd

# Honest variants: better (tree_coupling) and weaker (tree_rrs).
SCORES_TEST_OUT="${OUT}/variants.json" bash scripts/evaluate_test.sh eval/test/hardening/variants

# Shortcut programs (all must be scored as failed).
SCORES_TEST_OUT="${OUT}/shortcuts.json" bash scripts/evaluate_test.sh eval/test/hardening/shortcuts

# Baseline move: a degenerate program under the baseline's name; the fixed
# baseline value computed by the evaluator must not change.
SCORES_TEST_OUT="${OUT}/baseline_move.json" bash scripts/evaluate_test.sh eval/test/hardening/baseline_move

# Fixed truth: make the visible draft model identical to the target in
# data/synthetic_lm.py.  The train slice changes, the test slice must not.
cp data/synthetic_lm.py "${OUT}/synthetic_lm.py.orig"
trap 'cp "${OUT}/synthetic_lm.py.orig" data/synthetic_lm.py; rm -f "${OUT}/synthetic_lm.py.orig"' EXIT
python3 - <<'PY'
from pathlib import Path
p = Path("data/synthetic_lm.py")
s = p.read_text()
old = "    return p, q\n"
assert old in s
p.write_text(s.replace(old, "    return p, list(p)  # fixed-truth shortcut: perfect draft\n", 1))
PY
SCORES_TEST_OUT="${OUT}/fixed_truth_test.json" bash scripts/evaluate_test.sh checkpoints/hsd
python3 -m eval.train.evaluate checkpoints/hsd --output "${OUT}/fixed_truth_train.json"
cp "${OUT}/synthetic_lm.py.orig" data/synthetic_lm.py

python3 eval/test/hardening/rank.py
