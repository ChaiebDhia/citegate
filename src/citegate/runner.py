"""Async, resumable evaluation runner.

Results stream to JSONL as they complete. Re-running with the same ``out`` file skips finished
cases, so a crash or a rate-limit stop loses nothing. Concurrency is bounded by a semaphore, which
matters for I/O-bound LLM agents (CPU-bound agents simply run back to back).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

from citegate.agents import Agent, AgentAnswer
from citegate.corpus import Case, World
from citegate.metrics import CaseResult, score_case

BATCH = 512


def load_results(path: Path) -> dict[str, CaseResult]:
    done: dict[str, CaseResult] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result = CaseResult.from_dict(json.loads(line))
                done[result.case_id] = result
    return done


def _batches(items: Sequence[Case], size: int) -> Iterator[Sequence[Case]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def run_cases(
    agent: Agent,
    world: World,
    *,
    concurrency: int = 32,
    out: Path | None = None,
    limit: int | None = None,
) -> list[CaseResult]:
    chunks = world.chunk_map()
    cases = world.cases[:limit] if limit else world.cases
    done = load_results(out) if out else {}
    todo = [c for c in cases if c.id not in done]
    sem = asyncio.Semaphore(concurrency)

    async def one(case: Case) -> CaseResult:
        async with sem:
            start = time.perf_counter()
            error: str | None = None
            try:
                ans = await agent.answer(case.question)
            except Exception as exc:  # an agent crash is a failed case, not a crashed run
                ans, error = AgentAnswer(""), f"{type(exc).__name__}: {exc}"
            return score_case(case, ans, chunks, (time.perf_counter() - start) * 1000, error)

    handle = out.open("a", encoding="utf-8") if out else None
    try:
        for batch in _batches(todo, BATCH):
            for result in await asyncio.gather(*(one(c) for c in batch)):
                done[result.case_id] = result
                if handle:
                    handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            if handle:
                handle.flush()
    finally:
        if handle:
            handle.close()
    return [done[c.id] for c in cases]
