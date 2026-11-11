from __future__ import annotations

from citegate.corpus import World
from citegate.retrieval import BM25Index


def test_bm25_ranks_the_named_service_first(world: World, index: BM25Index) -> None:
    case = next(c for c in world.cases if c.id == "days-fr-s00001-days")
    assert index.search(case.question, 3)[0][0].id == "s00001-fr-fees"


def test_pruned_index_still_finds_the_right_service(world: World) -> None:
    pruned = BM25Index(world.chunks, max_postings=20)
    hits = pruned.search("How much is the fee for passport renewal (Aldoria)?", 4)
    assert hits and all("passport renewal" in h[0].service for h in hits)
