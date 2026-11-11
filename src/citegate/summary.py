"""Aggregate case results into per-group proportions with Wilson intervals."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from citegate.metrics import METRICS, CaseResult
from citegate.stats import Proportion, percentile, wilson


@dataclass(frozen=True, slots=True)
class Summary:
    agent: str
    n: int
    groups: dict[str, dict[str, Proportion]]
    latency_p50_ms: float
    latency_p95_ms: float


def summarise(agent: str, results: Sequence[CaseResult]) -> Summary:
    buckets: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        buckets["ALL"].append(r)
        buckets[f"kind:{r.kind}"].append(r)
        buckets[f"lang:{r.lang}"].append(r)
    groups = {
        name: {m: wilson(sum(bool(getattr(r, m)) for r in rs), len(rs)) for m in METRICS}
        for name, rs in sorted(buckets.items())
    }
    lat = [r.latency_ms for r in results]
    return Summary(agent, len(results), groups, percentile(lat, 50), percentile(lat, 95))
