from __future__ import annotations

import asyncio
from pathlib import Path

from citegate.agents import AgentAnswer, ExtractiveAgent
from citegate.corpus import World
from citegate.retrieval import BM25Index
from citegate.runner import load_results, run_cases


def test_runner_is_resumable(world: World, index: BM25Index, tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    first = asyncio.run(run_cases(ExtractiveAgent(index), world, out=out, limit=100))
    assert len(first) == 100 and len(load_results(out)) == 100

    class Boom:
        name = "boom"

        async def answer(self, question: str) -> AgentAnswer:
            raise AssertionError("finished cases must not be re-run")

    done = asyncio.run(run_cases(Boom(), world, out=out, limit=100))
    assert [r.case_id for r in done] == [r.case_id for r in first]


def test_agent_crash_is_a_failed_case(world: World) -> None:
    class Crash:
        name = "crash"

        async def answer(self, question: str) -> AgentAnswer:
            raise RuntimeError("rate limited")

    res = asyncio.run(run_cases(Crash(), world, limit=3))
    assert all(not r.passed and r.error and "rate limited" in r.error for r in res)


def test_reference_agent_is_perfect_on_golden_set(reference: list) -> None:  # type: ignore[type-arg]
    assert len(reference) == 744 and all(r.passed for r in reference)
