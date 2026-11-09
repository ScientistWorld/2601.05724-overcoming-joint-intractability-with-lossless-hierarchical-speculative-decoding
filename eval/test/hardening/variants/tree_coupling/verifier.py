"""Hardening variant 'tree_coupling' (tree verification with optimal_coupler at shared nodes,
HSD on single-draft chains).  Hidden from agents; used only by run_hardening.sh."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _hsd_ref import serve  # noqa: E402
from _tree import optimal_coupler, tree_verify  # noqa: E402

if __name__ == "__main__":
    serve(lambda request: tree_verify(request, optimal_coupler))
