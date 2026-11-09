"""Block verification baseline (Sun et al., 2024), verifier program.

Single-draft block verification: with p_0 = 1 and
p_i = min{p_{i-1} p(x_i)/q(x_i), 1}, the acceptance weight at position i < gamma is
    h_i = sum_x (p_i p(x|X_{1:i}) - q(x|X_{1:i}))_+ /
          (sum_x (p_i p(x|X_{1:i}) - q(x|X_{1:i}))_+ + 1 - p_i),
h_gamma = p_gamma, tau = max{i : eta_i <= h_i}, and the next token comes from
norm((p_tau p(.|X_{1:tau}) - q(.|X_{1:tau}))_+) (or a bonus token from p when
tau = gamma).  The paper reports no multi-draft extension of blockwise
verification, so with several drafts this baseline verifies only the first one.

Reports the exact outcome distribution (gym verifier protocol).
``scripts/baseline.sh`` copies it to ``checkpoints/blockwise/verifier.py``.
"""

from __future__ import annotations

import json
import sys


def block_chain(target_rows, draft_rows, draft):
    gamma = len(draft)
    vocab = len(target_rows[0])
    p_run = [1.0]
    for i in range(gamma):
        tok = draft[i]
        q_tok = draft_rows[i][tok]
        r = target_rows[i][tok] / q_tok if q_tok > 0.0 else 0.0
        p_run.append(min(p_run[-1] * r, 1.0))
    h = [1.0] * (gamma + 1)
    resid = [None] * (gamma + 1)
    for i in range(gamma):
        w = [max(p_run[i] * target_rows[i][x] - draft_rows[i][x], 0.0) for x in range(vocab)]
        num = sum(w)
        den = num + 1.0 - p_run[i]
        if i > 0:
            h[i] = 1.0 if den <= 0.0 else min(1.0, num / den)
        resid[i] = [v / num for v in w] if num > 0.0 else list(target_rows[i])
    h[gamma] = p_run[gamma]
    outcomes = []
    alive = 1.0
    for i in range(gamma, 0, -1):
        dist = list(target_rows[gamma]) if i == gamma else resid[i]
        outcomes.append((i, alive * h[i], dist))
        alive *= 1.0 - h[i]
    outcomes.append((0, alive, resid[0]))
    return outcomes


def verify(problem: dict) -> list[list]:
    draft = problem["drafts"][0]
    out: dict[tuple[int, int], float] = {}
    for tau, prob, dist in block_chain(problem["target_probs"][0], problem["draft_probs"][0], draft):
        for y, w in enumerate(dist):
            if prob > 0.0 and w > 0.0:
                out[(tau, y)] = out.get((tau, y), 0.0) + prob * w
    return [[0, tau, y, p] for (tau, y), p in sorted(out.items())]


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        sys.stdout.write(json.dumps({"id": request["id"], "outcomes": verify(request)}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
