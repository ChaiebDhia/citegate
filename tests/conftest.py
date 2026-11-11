from __future__ import annotations

import asyncio

import pytest

from citegate.agents import ExtractiveAgent
from citegate.corpus import World, build_world
from citegate.metrics import CaseResult
from citegate.retrieval import BM25Index
from citegate.runner import run_cases


@pytest.fixture(scope="session")
def world() -> World:
    return build_world()


@pytest.fixture(scope="session")
def index(world: World) -> BM25Index:
    return BM25Index(world.chunks)


@pytest.fixture(scope="session")
def reference(world: World, index: BM25Index) -> list[CaseResult]:
    return asyncio.run(run_cases(ExtractiveAgent(index), world))
