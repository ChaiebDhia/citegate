from __future__ import annotations

from citegate.retrieval import BM25Index, Chunk


def _chunks() -> list[Chunk]:
    return [
        Chunk("a", "passport renewal fee 40 dinars", "en", "passport renewal"),
        Chunk("b", "driving licence renewal fee 25 dinars", "en", "driving licence renewal"),
        Chunk("c", "tarif du renouvellement du passeport 40 dinars", "fr", "passeport"),
    ]


def test_bm25_ranks_the_matching_chunk_first() -> None:
    index = BM25Index(_chunks())
    assert index.search("passport fee")[0][0].id == "a"
    assert index.search("tarif passeport")[0][0].id == "c"


def test_bm25_is_deterministic_and_handles_unknown_terms() -> None:
    index = BM25Index(_chunks())
    assert index.search("zzzzqqq") == []
    assert index.search("renewal fee", 5) == index.search("renewal fee", 5)


def test_bm25_respects_k() -> None:
    assert len(BM25Index(_chunks()).search("renewal", 1)) == 1


def test_pruning_skips_common_terms_when_a_selective_term_exists() -> None:
    chunks = [Chunk(f"c{i}", f"fee item{i}", "en", "x") for i in range(30)]
    chunks.append(Chunk("rare", "fee zebra", "en", "x"))
    hits = BM25Index(chunks, max_postings=5).search("fee zebra", 10)
    assert [c.id for c, _ in hits] == ["rare"]
    assert len(BM25Index(chunks).search("fee zebra", 40)) == 31  # exact below the cap
