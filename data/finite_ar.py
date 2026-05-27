"""Small finite autoregressive distributions for speculative decoding tests.

The benchmark uses exact probability tables rather than pretrained LLMs so the
lossless distribution claim can be measured directly.  The problem remains the
same as in the paper: sample from a target autoregressive distribution p while
using a cheaper draft distribution q to propose blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product


VOCAB: tuple[str, ...] = ("A", "B", "C", "D", "E")


@dataclass(frozen=True)
class Scenario:
    name: str
    gamma: int
    max_len: int
    temperature: float
    description: str


SCENARIOS: dict[str, Scenario] = {
    "finite_ar_main": Scenario(
        name="finite_ar_main",
        gamma=4,
        max_len=4,
        temperature=1.0,
        description="finite AR target/draft pair, default draft length",
    ),
    "finite_ar_gamma3": Scenario(
        name="finite_ar_gamma3",
        gamma=3,
        max_len=4,
        temperature=1.0,
        description="finite AR target/draft pair, shorter draft length",
    ),
    "finite_ar_temp08": Scenario(
        name="finite_ar_temp08",
        gamma=4,
        max_len=4,
        temperature=0.8,
        description="finite AR target/draft pair, lower sampling temperature",
    ),
}


def _normalize(values: list[float]) -> dict[str, float]:
    total = sum(values)
    return {tok: val / total for tok, val in zip(VOCAB, values)}


def _apply_temperature(dist: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if abs(temperature - 1.0) < 1e-12:
        return dict(dist)
    values = [dist[tok] ** (1.0 / temperature) for tok in VOCAB]
    return _normalize(values)


def target_distribution(prefix: tuple[str, ...], temperature: float = 1.0) -> dict[str, float]:
    """Target model p(. | prefix)."""
    if not prefix:
        base = [0.36, 0.14, 0.22, 0.10, 0.18]
    else:
        last = VOCAB.index(prefix[-1])
        prev = VOCAB.index(prefix[-2]) if len(prefix) > 1 else 2
        step = len(prefix)
        values = []
        for i, _tok in enumerate(VOCAB):
            val = (
                0.18
                + 0.16 * (((i - last) % len(VOCAB)) == 1)
                + 0.12 * (((i + prev + step) % len(VOCAB)) == 0)
                + 0.08 * (i == last)
                + 0.03 * ((i * (step + 1) + prev) % 3)
            )
            values.append(val)
        base = values
    return _apply_temperature(_normalize(base), temperature)


def draft_distribution(prefix: tuple[str, ...], temperature: float = 1.0) -> dict[str, float]:
    """Draft model q(. | prefix), deliberately close to but not equal to p."""
    p = target_distribution(prefix, temperature=1.0)
    step = len(prefix)
    last = VOCAB.index(prefix[-1]) if prefix else 1
    values = []
    for i, tok in enumerate(VOCAB):
        bias = 0.72 + 0.22 * (((i + step) % len(VOCAB)) == last)
        bias += 0.18 * (((i - last) % len(VOCAB)) == 2)
        bias += 0.06 * ((i + 2 * step) % 2)
        values.append(max(1e-9, p[tok] * bias + 0.018 * (i + 1)))
    return _apply_temperature(_normalize(values), temperature)


def conditional(kind: str, prefix: tuple[str, ...], temperature: float = 1.0) -> dict[str, float]:
    if kind == "target":
        return target_distribution(prefix, temperature)
    if kind == "draft":
        return draft_distribution(prefix, temperature)
    raise ValueError(f"unknown distribution kind: {kind}")


def sample_from(dist: dict[str, float], rng) -> str:
    draw = rng.random()
    cumulative = 0.0
    last = VOCAB[-1]
    for tok in VOCAB:
        cumulative += dist[tok]
        if draw <= cumulative:
            return tok
    return last


def sequence_probability(
    tokens: tuple[str, ...], kind: str = "target", temperature: float = 1.0
) -> float:
    prefix: tuple[str, ...] = ()
    prob = 1.0
    for tok in tokens:
        dist = conditional(kind, prefix, temperature)
        prob *= dist[tok]
        prefix = prefix + (tok,)
    return prob


def branch_joint(
    prefix: tuple[str, ...], tok: str, kind: str, temperature: float = 1.0
) -> float:
    return sequence_probability(prefix, kind, temperature) * conditional(
        kind, prefix, temperature
    )[tok]


@lru_cache(maxsize=None)
def exact_sequence_distribution(max_len: int, temperature: float = 1.0) -> dict[str, float]:
    dist: dict[str, float] = {}
    for seq in product(VOCAB, repeat=max_len):
        key = "".join(seq)
        dist[key] = sequence_probability(seq, "target", temperature)
    return dist
