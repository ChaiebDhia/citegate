"""Small, dependency-free statistics used to gate on evidence rather than on noise."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

Z95 = 1.959964


@dataclass(frozen=True, slots=True)
class Proportion:
    k: int
    n: int
    p: float
    lo: float
    hi: float


def wilson(k: int, n: int, z: float = Z95) -> Proportion:
    """Wilson score interval. Well behaved at p=0 and p=1, unlike the normal approximation."""
    if n == 0:
        return Proportion(0, 0, 0.0, 0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Proportion(k, n, p, max(0.0, centre - half), min(1.0, centre + half))


def paired_delta(
    base: Sequence[bool], cand: Sequence[bool], z: float = Z95
) -> tuple[float, float, float]:
    """Paired difference in pass rate (candidate minus baseline) with a McNemar-style interval.

    Only discordant cases carry information: ``b`` regressions and ``c`` improvements.
    delta = (c - b) / n and var = (b + c - (c - b)^2 / n) / n^2. Pairing by case id removes
    case-difficulty noise, so small real regressions are detectable on a few hundred cases.
    """
    if len(base) != len(cand):
        raise ValueError("paired samples must have equal length")
    n = len(base)
    if n == 0:
        return 0.0, 0.0, 0.0
    b = sum(1 for x, y in zip(base, cand, strict=True) if x and not y)
    c = sum(1 for x, y in zip(base, cand, strict=True) if y and not x)
    delta = (c - b) / n
    se = math.sqrt(max(b + c - (c - b) ** 2 / n, 0.0)) / n
    return delta, delta - z * se, delta + z * se


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, ``q`` in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]
