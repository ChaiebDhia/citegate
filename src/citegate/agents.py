"""Agents under test: a deterministic reference agent and deliberately broken mutants.

The reference agent is an *extractive* baseline (retrieve, entity-link, copy the supporting
sentence, cite it). It needs no API key, so CI is hermetic. The mutants each break one property
(grounding, citation integrity, refusal, injection hygiene, freshness, language). They exist to
test the evals themselves: if a mutant passes the suite, the suite is blind to that failure.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from citegate.retrieval import BM25Index
from citegate.text import detect_lang, normalize, tokens


@dataclass(frozen=True, slots=True)
class AgentAnswer:
    text: str
    citations: tuple[str, ...] = ()
    refused: bool = False


class Agent(Protocol):
    """Anything that can answer a question. Plug your own agent in by implementing this."""

    name: str

    async def answer(self, question: str) -> AgentAnswer: ...


REFUSAL_TEXT = {
    "en": "I can't find this in the official guidance.",
    "fr": "Je ne trouve pas cette information dans les directives officielles.",
    "ar": "لا أجد هذه المعلومة في الإرشادات الرسمية.",
}

SLOT_ORDER = ("fee", "days", "age", "office")
_Q_CUES = {
    "fee": ("fee", "tarif", "رسوم"),
    "days": ("days", "jours", "يوم"),
    "age": ("age", "عمر"),
    "office": ("where", "ou", "اين"),
}
_S_CUES = {
    "fee": ("fee", "tarif", "رسوم"),
    "days": ("days", "jours", "يوم"),
    "age": ("years", "ans", "عمر"),
    "office": ("centre", "مركز"),
}
_SPLIT = re.compile(r"(?<=[.!?؟])\s+")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_INJECTION = re.compile(r"ignor\w*\b.*\binstructions|\bsystem\b|\bsysteme\b|تجاهل|النظام")


def refusal(question: str) -> AgentAnswer:
    return AgentAnswer(REFUSAL_TEXT[detect_lang(question)], (), True)


def _has_cue(toks: Sequence[str], cues: Iterable[str]) -> bool:
    present = set(toks)
    return any((cue in present) if cue.isascii() else any(cue in t for t in toks) for cue in cues)


class ExtractiveAgent:
    """Retrieve, link the entity named in the question, copy the supporting sentence."""

    name = "extractive"
    prefer_latest = True
    require_entity = True

    def __init__(self, index: BM25Index, k: int = 8) -> None:
        self.index, self.k = index, k

    async def answer(self, question: str) -> AgentAnswer:
        return self.respond(question)

    def respond(self, question: str) -> AgentAnswer:
        qtoks = tokens(question)
        slots = [s for s in SLOT_ORDER if _has_cue(qtoks, _Q_CUES[s])]
        hits = [chunk for chunk, _ in self.index.search(question, self.k)]
        if self.require_entity:
            asked = set(qtoks)
            hits = [c for c in hits if set(tokens(c.service, drop_stop=True)) <= asked]
        sentences: list[str] = []
        cited: list[str] = []
        for slot in slots:
            best = self._best_sentence(slot, hits) if hits else None
            if best is None:
                return refusal(question)
            sentences.append(best[0])
            cited.append(best[1])
        if not sentences:
            return refusal(question)
        return AgentAnswer(" ".join(sentences), tuple(dict.fromkeys(cited)))

    def _best_sentence(self, slot: str, hits: list) -> tuple[str, str] | None:  # type: ignore[type-arg]
        candidates: list[tuple[int, int, str, str]] = []
        for rank, chunk in enumerate(hits):
            found = _YEAR.search(chunk.text)
            year = int(found.group()) if found else 0
            for sentence in _SPLIT.split(chunk.text):
                if _INJECTION.search(normalize(sentence)):
                    continue
                if _has_cue(tokens(sentence), _S_CUES[slot]):
                    candidates.append((year, rank, sentence, chunk.id))
        if not candidates:
            return None
        pick = (
            max(candidates, key=lambda c: (c[0], -c[1]))
            if self.prefer_latest
            else min(candidates, key=lambda c: (c[0], c[1]))
        )
        return pick[2], pick[3]


class StalePicker(ExtractiveAgent):
    """Mutant: prefers the oldest guidance (no freshness handling)."""

    name = "mutant:stale_picker"
    prefer_latest = False


class NeverRefuser(ExtractiveAgent):
    """Mutant: skips entity linking, so it answers out-of-scope questions with a guess."""

    name = "mutant:never_refuser"
    require_entity = False


class OverRefuser(ExtractiveAgent):
    """Mutant: refuses everything."""

    name = "mutant:over_refuser"

    def respond(self, question: str) -> AgentAnswer:
        return refusal(question)


class Hallucinator(ExtractiveAgent):
    """Mutant: right sentence, wrong numbers, real citations."""

    name = "mutant:hallucinator"

    def respond(self, question: str) -> AgentAnswer:
        base = super().respond(question)
        if base.refused:
            return base
        bumped = re.sub(r"\d+", lambda m: str(int(m.group()) + 7), base.text)
        return AgentAnswer(bumped, base.citations)


class CitationForger(ExtractiveAgent):
    """Mutant: right answer, fabricated citation id."""

    name = "mutant:citation_forger"

    def respond(self, question: str) -> AgentAnswer:
        base = super().respond(question)
        return base if base.refused else AgentAnswer(base.text, ("doc-999",))


class NaiveCopy(ExtractiveAgent):
    """Mutant: pastes the top-3 retrieved chunks verbatim (injection and stale text included)."""

    name = "mutant:naive_copy"

    def respond(self, question: str) -> AgentAnswer:
        top = [chunk for chunk, _ in self.index.search(question, 3)]
        if not top:
            return refusal(question)
        return AgentAnswer(" ".join(c.text for c in top), tuple(c.id for c in top))


class LanguageDrift(ExtractiveAgent):
    """Mutant: answers French and Arabic questions in English."""

    name = "mutant:language_drift"

    def respond(self, question: str) -> AgentAnswer:
        base = super().respond(question)
        if base.refused or detect_lang(question) == "en":
            return base
        nums = re.findall(r"\d+", normalize(base.text))
        value = nums[0] if nums else "available at the service centre"
        return AgentAnswer(f"The answer is {value}.", base.citations)


# name -> (class, metric that must notice the defect)
MUTANTS: dict[str, tuple[type[ExtractiveAgent], str]] = {
    "stale_picker": (StalePicker, "clean"),
    "never_refuser": (NeverRefuser, "refusal_ok"),
    "over_refuser": (OverRefuser, "correct"),
    "hallucinator": (Hallucinator, "faithful"),
    "citation_forger": (CitationForger, "citations_valid"),
    "naive_copy": (NaiveCopy, "clean"),
    "language_drift": (LanguageDrift, "lang_ok"),
}
