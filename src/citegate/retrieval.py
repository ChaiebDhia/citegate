"""A small, dependency-free BM25 index with a deterministic ranking."""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from citegate.text import tokens


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable passage. ``service`` is the entity the passage is about."""

    id: str
    text: str
    lang: str
    service: str


class BM25Index:
    """Okapi BM25 over normalised, stop-word-free tokens (inverted index, O(postings))."""

    def __init__(
        self,
        chunks: Iterable[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
        max_postings: int = 2000,
    ) -> None:
        """``max_postings`` caps work per query: terms with longer posting lists are skipped when
        a rarer query term exists. Results are exact while the corpus has <= max_postings chunks.
        """
        self.chunks: list[Chunk] = list(chunks)
        self.k1, self.b, self.max_postings = k1, b, max_postings
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths: list[int] = []
        for idx, chunk in enumerate(self.chunks):
            toks = tokens(chunk.text, drop_stop=True)
            self._lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                self._postings[term].append((idx, tf))
        n = max(len(self.chunks), 1)
        self._avgdl = (sum(self._lengths) / n) or 1.0
        self._idf = {
            term: math.log(1 + (n - len(post) + 0.5) / (len(post) + 0.5))
            for term, post in self._postings.items()
        }

    def search(self, query: str, k: int = 8) -> list[tuple[Chunk, float]]:
        scores: dict[int, float] = defaultdict(float)
        terms = [t for t in set(tokens(query, drop_stop=True)) if t in self._idf]
        selective = [t for t in terms if len(self._postings[t]) <= self.max_postings]
        for term in sorted(selective or terms):
            idf = self._idf[term]
            for idx, tf in self._postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self._lengths[idx] / self._avgdl)
                scores[idx] += idf * tf * (self.k1 + 1) / (tf + norm)
        top = heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
        return [(self.chunks[idx], score) for idx, score in top]
