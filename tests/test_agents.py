from __future__ import annotations

import asyncio

from citegate.agents import MUTANTS, REFUSAL_TEXT, ExtractiveAgent, StalePicker
from citegate.corpus import World
from citegate.retrieval import BM25Index


def _case(world: World, case_id: str):  # type: ignore[no-untyped-def]
    return next(c for c in world.cases if c.id == case_id)


def test_reference_answers_with_the_supporting_sentence_and_citation(
    world: World, index: BM25Index
) -> None:
    case = _case(world, "fee-en-s00001-fee")
    answer = asyncio.run(ExtractiveAgent(index).answer(case.question))
    assert case.values[0] in answer.text
    assert answer.citations == ("s00001-en-fees",) and not answer.refused


def test_reference_refuses_out_of_scope_in_the_question_language(
    world: World, index: BM25Index
) -> None:
    for lang in ("en", "fr", "ar"):
        case = _case(world, f"oos-{lang}-0-fee")
        answer = ExtractiveAgent(index).respond(case.question)
        assert answer.refused and answer.text == REFUSAL_TEXT[lang]


def test_reference_ignores_planted_instructions(world: World, index: BM25Index) -> None:
    for lang in ("en", "fr", "ar"):
        case = _case(world, f"injection-{lang}-s00000-fee")
        answer = ExtractiveAgent(index).respond(case.question)
        assert "PWNED" not in answer.text.upper() and case.values[0] in answer.text


def test_reference_prefers_current_guidance_and_stale_picker_does_not(
    world: World, index: BM25Index
) -> None:
    case = _case(world, "stale-en-s00003-fee")
    assert case.forbidden[0] not in ExtractiveAgent(index).respond(case.question).text
    assert case.forbidden[0] in StalePicker(index).respond(case.question).text


def test_multi_fact_questions_cite_every_supporting_chunk(world: World, index: BM25Index) -> None:
    case = _case(world, "multihop-ar-s00002")
    answer = ExtractiveAgent(index).respond(case.question)
    assert set(answer.citations) == {"s00002-ar-fees", "s00002-ar-elig"}


def test_mutant_registry_names_the_metric_that_must_catch_each_defect() -> None:
    assert set(MUTANTS) == {
        "stale_picker",
        "never_refuser",
        "over_refuser",
        "hallucinator",
        "citation_forger",
        "naive_copy",
        "language_drift",
    }
