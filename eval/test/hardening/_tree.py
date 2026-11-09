"""Hardening variants for the multi-draft headroom check (hidden from agents).

Tree verification over i.i.d. drafts: at a node shared by several drafts the
next token is chosen by a lossless multi-draft coupling of the drafts' next
tokens with the target row; once a single draft remains, the rest of its chain
is verified with HSD (Algorithm 2).  ``optimal_coupler`` uses a maximum-flow
coupling of p with the multiset of n i.i.d. draft tokens (the optimal
single-token multi-draft acceptance); ``rrs_coupler`` uses recursive reject
sampling with replacement.  With one draft both reduce exactly to HSD.
"""

from __future__ import annotations

import itertools
from collections import deque

from _hsd_ref import hsd_chain

_FLOW_CACHE: dict = {}


def _norm(w):
    s = sum(w)
    return [v / s for v in w] if s > 0.0 else None


def rrs_coupler(P, Q, tokens):
    out = {}
    stay = 1.0
    cur = list(P)
    for x in tokens:
        if stay <= 0.0:
            break
        acc = min(1.0, cur[x] / Q[x]) if Q[x] > 0.0 else 0.0
        out[x] = out.get(x, 0.0) + stay * acc
        stay *= 1.0 - acc
        nxt = _norm([max(cur[y] - Q[y], 0.0) for y in range(len(P))])
        if nxt is None:
            stay = 0.0
            break
        cur = nxt
    if stay > 0.0:
        for y, v in enumerate(cur):
            if v > 0.0:
                out[y] = out.get(y, 0.0) + stay * v
    return out


def _max_flow(P, Q, n):
    key = (tuple(P), tuple(Q), n)
    if key in _FLOW_CACHE:
        return _FLOW_CACHE[key]
    V = len(P)
    tuples = list(itertools.product(range(V), repeat=n))
    supply = {}
    for t in tuples:
        w = 1.0
        for x in t:
            w *= Q[x]
        supply[t] = w
    cap_t = dict(supply)
    cap_y = list(P)
    flow = {t: {} for t in tuples}
    tiny = 1e-15
    while True:
        parent = {}
        queue = deque()
        for t in tuples:
            if cap_t[t] > tiny:
                parent[("t", t)] = None
                queue.append(("t", t))
        sink = None
        while queue and sink is None:
            node = queue.popleft()
            if node[0] == "t":
                for y in sorted(set(node[1])):
                    nxt = ("y", y)
                    if nxt not in parent:
                        parent[nxt] = node
                        queue.append(nxt)
            else:
                y = node[1]
                if cap_y[y] > tiny:
                    sink = node
                    break
                for t in tuples:
                    if flow[t].get(y, 0.0) > tiny and ("t", t) not in parent:
                        parent[("t", t)] = node
                        queue.append(("t", t))
        if sink is None:
            break
        path = []
        node = sink
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        bottleneck = min(cap_t[path[0][1]], cap_y[path[-1][1]])
        for a, b in zip(path, path[1:]):
            if a[0] == "y":
                bottleneck = min(bottleneck, flow[b[1]].get(a[1], 0.0))
        cap_t[path[0][1]] -= bottleneck
        cap_y[path[-1][1]] -= bottleneck
        for a, b in zip(path, path[1:]):
            if a[0] == "t":
                flow[a[1]][b[1]] = flow[a[1]].get(b[1], 0.0) + bottleneck
            else:
                flow[b[1]][a[1]] -= bottleneck
    inflow = [P[y] - cap_y[y] for y in range(V)]
    resid = _norm([max(P[y] - inflow[y], 0.0) for y in range(V)])
    _FLOW_CACHE[key] = (flow, supply, resid)
    return _FLOW_CACHE[key]


def optimal_coupler(P, Q, tokens):
    flow, supply, resid = _max_flow(P, Q, len(tokens))
    t = tuple(tokens)
    out = {}
    used = 0.0
    if supply[t] > 0.0:
        for y, v in flow[t].items():
            if v > 0.0:
                out[y] = v / supply[t]
                used += v / supply[t]
    rest = 1.0 - used
    if rest > 0.0:
        if resid is None:
            total = sum(out.values())
            return {y: v / total for y, v in out.items()}
        for y, v in enumerate(resid):
            if v > 0.0:
                out[y] = out.get(y, 0.0) + rest * v
    return out


def tree_verify(problem, coupler):
    drafts = problem["drafts"]
    P = problem["target_probs"]
    Q = problem["draft_probs"]
    gamma = len(drafts[0])
    merged = {}

    def emit(k, tau, y, prob):
        if prob > 0.0:
            merged[(k, tau, y)] = merged.get((k, tau, y), 0.0) + prob

    def rec(depth, idxs, prob):
        k0 = idxs[0]
        if depth == gamma:
            for y, v in enumerate(P[k0][gamma]):
                emit(k0, gamma, y, prob * v)
            return
        if len(idxs) == 1:
            outcomes, rows = hsd_chain(P[k0][depth:], Q[k0][depth:], drafts[k0][depth:])
            for t, p_t, dist in outcomes:
                for y, v in enumerate(dist):
                    emit(k0, depth + t, y, prob * p_t * v)
            return
        tokens = [drafts[k][depth] for k in idxs]
        for y, v in coupler(P[k0][depth], Q[k0][depth], tokens).items():
            if v <= 0.0:
                continue
            sub = [k for k in idxs if drafts[k][depth] == y]
            if sub:
                rec(depth + 1, sub, prob * v)
            else:
                emit(k0, depth, y, prob * v)

    rec(0, list(range(len(drafts))), 1.0)
    return [[k, tau, y, p] for (k, tau, y), p in sorted(merged.items())]
