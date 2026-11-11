"""Regression gate: fail the build on evidence, not on noise."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from citegate.metrics import CaseResult
from citegate.stats import paired_delta
from citegate.summary import Summary

GATED_METRICS = (
    "passed",
    "correct",
    "faithful",
    "clean",
    "citations_valid",
    "refusal_ok",
    "lang_ok",
)


@dataclass(frozen=True, slots=True)
class GateConfig:
    min_pass_lower_bound: float = 0.95  # Wilson 95 % lower bound of the overall pass rate
    min_group_pass: float = 0.90  # point estimate floor for every kind and language
    tolerance: float = 0.01  # largest tolerated drop in any gated metric vs baseline


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass(slots=True)
class GateReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks)

    def markdown(self) -> str:
        rows = "\n".join(
            f"| {'PASS' if c.ok else '**FAIL**'} | `{c.name}` | {c.detail} |" for c in self.checks
        )
        verdict = "PASSED" if self.passed else "FAILED"
        return f"### citegate gate: {verdict}\n\n| | check | detail |\n|---|---|---|\n{rows}\n"


def baseline_from_results(agent: str, results: Sequence[CaseResult]) -> dict[str, object]:
    cases = {r.case_id: {m: int(bool(getattr(r, m))) for m in GATED_METRICS} for r in results}
    return {"version": 1, "agent": agent, "cases": cases}


def save_baseline(path: Path, agent: str, results: Sequence[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = baseline_from_results(agent, results)
    path.write_text(json.dumps(payload, sort_keys=True, indent=0) + "\n", encoding="utf-8")


def evaluate_gate(
    summary: Summary,
    results: Sequence[CaseResult],
    baseline: dict[str, dict[str, int]],
    config: GateConfig | None = None,
) -> GateReport:
    config = config or GateConfig()
    report = GateReport()
    overall = summary.groups["ALL"]["passed"]
    report.checks.append(
        Check(
            "floor:overall",
            overall.lo >= config.min_pass_lower_bound,
            f"pass {overall.p:.1%} (95% CI {overall.lo:.1%} to {overall.hi:.1%}, n={overall.n}), "
            f"need lower bound >= {config.min_pass_lower_bound:.0%}",
        )
    )
    for name, metrics in summary.groups.items():
        if name != "ALL":
            p = metrics["passed"]
            report.checks.append(
                Check(
                    f"floor:{name}",
                    p.p >= config.min_group_pass,
                    f"pass {p.p:.1%} ({p.k}/{p.n}), need >= {config.min_group_pass:.0%}",
                )
            )
    paired = [r for r in results if r.case_id in baseline]
    if not paired:
        report.checks.append(Check("baseline:compatible", False, "no case ids in common"))
        return report
    for metric in GATED_METRICS:
        base = [bool(baseline[r.case_id][metric]) for r in paired]
        cand = [bool(getattr(r, metric)) for r in paired]
        delta, lo, hi = paired_delta(base, cand)
        significant = hi < 0
        ok = not (significant or delta < -config.tolerance)
        report.checks.append(
            Check(
                f"regression:{metric}",
                ok,
                f"delta {delta:+.2%} vs baseline (95% CI {lo:+.2%} to {hi:+.2%}, n={len(paired)})",
            )
        )
    return report
