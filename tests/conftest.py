from __future__ import annotations

import pytest

from citegate.corpus import World, build_world
from citegate.retrieval import BM25Index


@pytest.fixture(scope="session")
def world() -> World:
    return build_world()


@pytest.fixture(scope="session")
def index(world: World) -> BM25Index:
    return BM25Index(world.chunks)
