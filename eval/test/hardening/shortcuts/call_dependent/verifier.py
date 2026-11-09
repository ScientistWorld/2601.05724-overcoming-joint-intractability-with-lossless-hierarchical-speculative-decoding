"""Shortcut: stateful program that switches between HSD and an inflated
acceptance rule depending on how many calls it has served (answers depend on
call order, not only on the problem)."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _hsd_ref  # noqa: E402
from _hsd_ref import serve, verify  # noqa: E402

_orig = _hsd_ref.hsd_chain
_count = [0]

def answer(req):
    _count[0] += 1
    if _count[0] % 3 == 0:
        g = len(req["drafts"][0])
        return [[0, g, y, p] for y, p in enumerate(req["target_probs"][0][g]) if p > 0]
    return verify(req)

if __name__ == "__main__":
    serve(answer)
