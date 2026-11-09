"""Shortcut: HSD with every backward acceptance probability inflated by 2%
(slightly lossy, slightly faster).  Old analogue: biased_answers_shortcut."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _hsd_ref  # noqa: E402
from _hsd_ref import serve, verify  # noqa: E402

_orig = _hsd_ref.hsd_chain

def inflated(target_rows, draft_rows, draft, root_target=None):
    outcomes, rows = _orig(target_rows, draft_rows, draft, root_target)
    # recover h_t from the scan and inflate it
    hs = {}
    survive = 1.0
    for t, p, _ in outcomes:
        if t > 0:
            hs[t] = p / survive if survive > 0 else 1.0
            survive -= p
    dists = {t: d for t, _, d in outcomes}
    new = []
    survive = 1.0
    for t in sorted(hs, reverse=True):
        h = min(1.0, hs[t] * 1.02)
        new.append((t, survive * h, dists[t]))
        survive *= 1.0 - h
    new.append((0, survive, dists[0]))
    return new, rows

_hsd_ref.hsd_chain = inflated

if __name__ == "__main__":
    serve(verify)
