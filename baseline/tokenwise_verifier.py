"""Token-wise speculative verification baseline (verifier program).

* single draft: Leviathan et al. (2023): accept x_t with probability
  min{1, p(x_t)/q(x_t)} left to right; at the first rejection emit one token
  from norm(max(p - q, 0)); if all tokens pass emit a bonus token from p.
* several drafts: the paper's multi-draft baseline, recursive reject sampling
  with replacement (Yang et al., 2024) on i.i.d. drafts: drafts are verified in
  order, a draft is only used if it extends the accepted prefix, and after a
  rejection the residual becomes the target at the rejection node.

Reports the exact outcome distribution (gym verifier protocol, one JSON line
in, one JSON line out).  ``scripts/baseline.sh`` copies it to
``checkpoints/tokenwise/verifier.py``.
"""

from __future__ import annotations

import json
import sys


def _residual(p: list[float], q: list[float]) -> list[float]:
    w = [max(a - b, 0.0) for a, b in zip(p, q)]
    s = sum(w)
    return [v / s for v in w] if s > 0.0 else list(p)


def tokenwise_chain(target_rows, draft_rows, draft, root_target=None):
    gamma = len(draft)
    rows = list(target_rows)
    if root_target is not None:
        rows[0] = root_target
    outcomes = []
    alive = 1.0
    for t in range(gamma):
        tok = draft[t]
        q_tok = draft_rows[t][tok]
        a = min(1.0, rows[t][tok] / q_tok) if q_tok > 0.0 else 0.0
        outcomes.append((t, alive * (1.0 - a), _residual(rows[t], draft_rows[t])))
        alive *= a
    outcomes.append((gamma, alive, list(rows[gamma])))
    return outcomes


def verify(problem: dict) -> list[list]:
    drafts = problem["drafts"]
    target = problem["target_probs"]
    draft_probs = problem["draft_probs"]
    num_drafts = len(drafts)
    gamma = len(drafts[0])
    merged: dict[tuple[int, int, int], float] = {}

    def emit(k, tau, dist, prob):
        for y, w in enumerate(dist):
            if w > 0.0 and prob > 0.0:
                merged[(k, tau, y)] = merged.get((k, tau, y), 0.0) + prob * w

    def stage(k, tau, path, root_target, prob, last_k):
        while k < num_drafts and tuple(drafts[k][:tau]) != path:
            k += 1
        if k >= num_drafts:
            emit(last_k, tau, root_target, prob)
            return
        for t, p_t, dist in tokenwise_chain(
            target[k][tau:], draft_probs[k][tau:], drafts[k][tau:], root_target
        ):
            if p_t <= 0.0:
                continue
            new_tau = tau + t
            if new_tau == gamma:
                emit(k, gamma, dist, prob * p_t)
            else:
                stage(k + 1, new_tau, tuple(drafts[k][:new_tau]), dist, prob * p_t, k)

    stage(0, 0, (), None, 1.0, 0)
    return [[k, tau, y, p] for (k, tau, y), p in sorted(merged.items())]


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
