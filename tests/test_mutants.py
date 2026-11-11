"""The evals are tested too: every deliberately broken agent must be caught."""

from __future__ import annotations

import asyncio

import pytest

from citegate.agents import MUTANTS
from citegate.corpus import World
from citegate.gate import baseline_from_results, evaluate_gate
from citegate.metrics import CaseResult
from citegate.retrieval import BM25Index
from citegate.runner import run_cases
from citegate.stats import paired_delta
from citegate.summary import summarise


def test_reference_agent_is_perfect_on_golden_set(reference: list[CaseResult]) -> None:
    assert all(r.passed for r in reference)


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_caught(
    name: str, world: World, index: BM25Index, reference: list[CaseResult]
) -> None:
    cls, metric = MUTANTS[name]
    ref = reference
    mut = asyncio.run(run_cases(cls(index), world))
    delta, _, hi = paired_delta(
        [bool(getattr(r, metric)) for r in ref], [bool(getattr(r, metric)) for r in mut]
    )
    assert delta < 0 and hi < 0, f"{name}: metric {metric} did not detect the defect"
    base = baseline_from_results("extractive", ref)["cases"]
    assert not evaluate_gate(summarise(name, mut), mut, base).passed  # type: ignore[arg-type]
