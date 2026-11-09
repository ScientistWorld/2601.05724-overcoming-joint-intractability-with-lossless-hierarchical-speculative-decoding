"""Shortcut: HSD without the capping of prefix ratios (the paper's Table A.3
ablation: slightly higher acceptance, distribution not preserved)."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _hsd_ref  # noqa: E402
from _hsd_ref import serve, verify  # noqa: E402


def hsd_chain_uncapped(target_rows, draft_rows, draft, root_target=None):
    gamma = len(draft)
    rows = list(target_rows)
    if root_target is not None:
        rows[0] = root_target
    vocab = len(rows[0])
    ratio = [1.0]
    for i in range(gamma):
        q_tok = draft_rows[i][draft[i]]
        ratio.append(ratio[-1] * rows[i][draft[i]] / q_tok if q_tok > 0 else 0.0)
    accept = [1.0] * (gamma + 1)
    residual = [None] * (gamma + 1)
    for t in range(gamma):
        scale = ratio[t]  # uncapped joint ratio
        d_pq = sum(max(scale * rows[t][y] - draft_rows[t][y], 0.0) for y in range(vocab))
        d_qp = sum(max(draft_rows[t][y] - scale * rows[t][y], 0.0) for y in range(vocab))
        if t > 0:
            accept[t] = 1.0 if d_qp <= 0 else min(1.0, d_pq / d_qp)
        w = [max(scale * rows[t][y] - draft_rows[t][y], 0.0) for y in range(vocab)]
        residual[t] = _hsd_ref._normalized(w) or list(rows[t])
    accept[gamma] = min(1.0, ratio[gamma])
    outcomes = []
    survive = 1.0
    for t in range(gamma, 0, -1):
        outcomes.append((t, survive * accept[t], list(rows[gamma]) if t == gamma else residual[t]))
        survive *= 1.0 - accept[t]
    outcomes.append((0, survive, residual[0]))
    return outcomes, rows

_hsd_ref.hsd_chain = hsd_chain_uncapped

if __name__ == "__main__":
    serve(verify)
