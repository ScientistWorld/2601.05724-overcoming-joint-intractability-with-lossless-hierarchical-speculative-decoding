"""Shortcut: replaces the shipped tokenwise baseline program with a degenerate
one (old analogue: baseline_move_shortcut).  The fixed baseline value is
computed inside the evaluator and must not move."""
import json
import sys

for line in sys.stdin:
    req = json.loads(line)
    sys.stdout.write(json.dumps({"id": req["id"], "outcomes": [[0, 0, 0, 1.0]]}) + "\n")
    sys.stdout.flush()
