"""Shortcut: answers with no outcomes.  Old analogue: empty_shortcut."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _hsd_ref  # noqa: E402
from _hsd_ref import serve, verify  # noqa: E402


if __name__ == "__main__":
    serve(lambda req: [])
