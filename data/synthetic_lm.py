"""Synthetic target/draft language-model pairs for the verification benchmark.

Every node (context + continuation) of an instance has a target next-token
distribution p(.|node) and a draft distribution q(.|node).  Rows are produced
deterministically from a hash of (salt, instance seed, family, node):

    z ~ N(0, logit_scale^2)  per token           (target logits)
    e ~ N(0, 1)              per token           (draft disagreement)
    p = softmax(z / T),  q = softmax((shrink * z + noise * e) / T)

``shrink < 1`` makes the draft flatter than the target (a smaller model),
``noise`` controls how often draft and target disagree.  Rows at different
nodes are independent, so nothing about an unseen branch can be inferred from
the rows a verifier is shown.

This module defines the *visible* (train) instances.  The held-out test
instances are drawn from the same families with different, hidden seeds.
"""

from __future__ import annotations

import hashlib
import math
import random

TRAIN_SALT = "hsd-gym-visible-train-v2"
TRAIN_SEED = 20260527

# Content regimes (analogues of the paper's datasets: low-entropy maths/code
# continuations vs. high-entropy summarisation).
REGIMES: dict[str, float] = {
    "peaked": 3.0,
    "medium": 2.0,
    "flat": 1.2,
}

# Draft/target pairings (analogues of pairing one small draft with targets of
# different sizes).
DRAFT_PAIRS: dict[str, tuple[float, float]] = {
    "aligned": (0.85, 0.6),
    "misaligned": (0.7, 1.1),
}


def _rng(*key) -> random.Random:
    digest = hashlib.sha256(repr(key).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _softmax(logits: list[float], temperature: float) -> list[float]:
    top = max(logits)
    exps = [math.exp((v - top) / temperature) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


def node_distributions(
    salt: str,
    seed: int,
    regime: str,
    pair: str,
    vocab: int,
    temperature: float,
    node: tuple[int, ...],
) -> tuple[list[float], list[float]]:
    """Return (p(.|node), q(.|node)) for one node of an instance."""
    rng = _rng(salt, seed, regime, pair, vocab, node)
    scale = REGIMES[regime]
    shrink, noise = DRAFT_PAIRS[pair]
    z = [rng.gauss(0.0, 1.0) * scale for _ in range(vocab)]
    e = [rng.gauss(0.0, 1.0) for _ in range(vocab)]
    p = _softmax(z, temperature)
    q = _softmax([shrink * a + noise * b for a, b in zip(z, e)], temperature)
    return p, q


def experiment_configs() -> dict[str, list[dict]]:
    """Benchmark settings shared by the train and test slices."""
    configs: dict[str, list[dict]] = {
        "single_draft_main": [],
        "single_draft_temperature": [],
        "single_draft_length": [],
        "multi_draft": [],
    }
    for regime in REGIMES:
        for pair in DRAFT_PAIRS:
            configs["single_draft_main"].append(
                dict(regime=regime, pair=pair, vocab=6, gamma=4, drafts=1, temperature=1.0)
            )
        for temperature in (0.6, 0.8):
            configs["single_draft_temperature"].append(
                dict(regime=regime, pair="aligned", vocab=6, gamma=4, drafts=1, temperature=temperature)
            )
        for gamma in (2, 3, 4, 5):
            configs["single_draft_length"].append(
                dict(regime=regime, pair="aligned", vocab=5, gamma=gamma, drafts=1, temperature=1.0)
            )
        for vocab, drafts, gamma in ((3, 2, 4), (4, 2, 3), (3, 3, 3)):
            for pair in DRAFT_PAIRS:
                configs["multi_draft"].append(
                    dict(regime=regime, pair=pair, vocab=vocab, gamma=gamma, drafts=drafts, temperature=1.0)
                )
    return configs


def sample_contexts(salt: str, seed: int, config: dict, count: int) -> list[tuple[int, ...]]:
    """Contexts reached by sampling the target model from an empty prompt."""
    contexts = []
    for index in range(count):
        rng = _rng(salt, seed, "context", sorted(config.items()), index)
        length = 2 + rng.randrange(5)
        ctx: tuple[int, ...] = (index,)  # distinct prompt token per context
        for _ in range(length):
            p, _q = node_distributions(
                salt, seed, config["regime"], config["pair"], config["vocab"],
                config["temperature"], ctx,
            )
            u = rng.random()
            acc = 0.0
            tok = len(p) - 1
            for i, v in enumerate(p):
                acc += v
                if u <= acc:
                    tok = i
                    break
            ctx = ctx + (tok,)
        contexts.append(ctx)
    return contexts


# Visible contexts per configuration (the hidden test slice uses more).
TRAIN_CONTEXTS_PER_CONFIG: dict[str, int] = {
    "single_draft_main": 3,
    "single_draft_temperature": 3,
    "single_draft_length": 3,
    "multi_draft": 1,
}
