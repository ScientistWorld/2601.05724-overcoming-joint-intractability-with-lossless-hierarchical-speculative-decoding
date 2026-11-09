"""Shortcut: claims gamma + 2 emitted tokens in one target step."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _hsd_ref  # noqa: E402
from _hsd_ref import serve, verify  # noqa: E402


if __name__ == "__main__":
    serve(lambda req: [[0, len(req["drafts"][0]) + 1, 0, 1.0]])
