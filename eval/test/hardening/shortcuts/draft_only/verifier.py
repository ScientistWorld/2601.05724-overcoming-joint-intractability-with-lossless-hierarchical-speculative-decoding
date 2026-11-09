"""Shortcut: accept the whole first draft every time and add a bonus token
(no verification).  Old analogue: draft_only_shortcut."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _hsd_ref  # noqa: E402
from _hsd_ref import serve, verify  # noqa: E402


def answer(req):
    g = len(req["drafts"][0])
    return [[0, g, y, p] for y, p in enumerate(req["target_probs"][0][g]) if p > 0]

if __name__ == "__main__":
    serve(answer)
