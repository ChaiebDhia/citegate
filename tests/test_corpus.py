from __future__ import annotations

from collections import Counter

from citegate.corpus import build_world


def test_world_is_deterministic_and_seed_sensitive() -> None:
    assert build_world(12, 1) == build_world(12, 1)
    assert build_world(12, 1) != build_world(12, 2)


def test_golden_set_composition(world: object) -> None:
    kinds = Counter(c.kind for c in build_world().cases)
    assert kinds["out_of_scope"] == 24
    assert kinds["injection"] == 24
    assert kinds["stale"] == 24
    assert sum(kinds.values()) == 744


def test_case_ids_unique_and_gold_chunks_exist() -> None:
    w = build_world()
    assert len({c.id for c in w.cases}) == len(w.cases)
    ids = set(w.chunk_map())
    assert all(g in ids for c in w.cases for group in c.gold_chunks for g in group)


def test_scaled_world_uses_generated_zone_names() -> None:
    w = build_world(60)
    assert any("Zone" in c.service for c in w.chunks)


def test_world_sample_is_stratified_and_deterministic() -> None:
    world = build_world()
    sample = world.sample(24)
    assert len(sample.cases) == 24
    assert len({c.kind for c in sample.cases}) == 8
    assert {c.lang for c in sample.cases} == {"en", "fr", "ar"}
    assert sample == world.sample(24)
    assert world.sample(10_000) is world
