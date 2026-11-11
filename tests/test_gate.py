from __future__ import annotations

import asyncio
import json
from pathlib import Path

from citegate.agents import Hallucinator
from citegate.corpus import World
from citegate.gate import GateConfig, baseline_from_results, evaluate_gate
from citegate.metrics import CaseResult
from citegate.retrieval import BM25Index
from citegate.runner import run_cases
from citegate.summary import summarise


def test_gate_passes_reference_and_fails_on_regression(
    world: World, index: BM25Index, reference: list[CaseResult]
) -> None:
    base = baseline_from_results("extractive", reference)["cases"]
    ok = evaluate_gate(summarise("extractive", reference), reference, base)  # type: ignore[arg-type]
    assert ok.passed and "PASSED" in ok.markdown()

    bad = asyncio.run(run_cases(Hallucinator(index), world))
    report = evaluate_gate(summarise("bad", bad), bad, base)  # type: ignore[arg-type]
    assert not report.passed
    assert any(c.name == "regression:faithful" and not c.ok for c in report.checks)
    assert "FAILED" in report.markdown()


def test_gate_handles_incompatible_baseline(reference: list[CaseResult]) -> None:
    head = reference[:10]
    report = evaluate_gate(summarise("x", head), head, {}, GateConfig())
    assert not report.passed and report.checks[-1].name == "baseline:compatible"


def test_committed_baseline_matches_current_reference(reference: list[CaseResult]) -> None:
    committed = json.loads(Path("baselines/reference.json").read_text(encoding="utf-8"))["cases"]
    assert committed == baseline_from_results("extractive", reference)["cases"]
