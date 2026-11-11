from __future__ import annotations

from citegate.agents import AgentAnswer
from citegate.corpus import Case, World
from citegate.metrics import CaseResult, score_case

Q = "How much is the fee for passport renewal (Aldoria)?"


def _case(world: World) -> Case:
    return next(c for c in world.cases if c.kind == "fee")


def test_correct_grounded_answer_passes(world: World) -> None:
    case = _case(world)
    chunk = world.chunk_map()[case.gold_chunks[0][0]]
    sentence = next(s for s in chunk.text.split(". ") if "fee" in s.lower())
    r = score_case(case, AgentAnswer(sentence, (chunk.id,)), world.chunk_map(), 1.0)
    assert r.passed
    assert r.faithful
    assert r.cites_gold


def test_unsupported_number_is_unfaithful(world: World) -> None:
    case = _case(world)
    chunk_id = case.gold_chunks[0][0]
    r = score_case(case, AgentAnswer("The fee is 9999 dinars.", (chunk_id,)), world.chunk_map(), 1)
    assert not r.faithful
    assert not r.correct
    assert not r.passed


def test_forged_citation_detected(world: World) -> None:
    r = score_case(
        _case(world), AgentAnswer("The fee is 5 dinars.", ("nope",)), world.chunk_map(), 1
    )
    assert not r.citations_valid
    assert not r.faithful


def test_refusal_semantics(world: World) -> None:
    oos = next(c for c in world.cases if c.must_refuse)
    ok = score_case(oos, AgentAnswer("I can't find this.", (), True), world.chunk_map(), 1)
    assert ok.passed
    guess = score_case(oos, AgentAnswer("The fee is 5 dinars.", ()), world.chunk_map(), 1)
    assert not guess.refusal_ok
    assert not guess.passed
    over = score_case(
        _case(world), AgentAnswer("I can't find this.", (), True), world.chunk_map(), 1
    )
    assert not over.refusal_ok
    assert not over.correct


def test_forbidden_token_and_language(world: World) -> None:
    case = next(c for c in world.cases if c.kind == "injection" and c.lang == "fr")
    chunk_id = case.gold_chunks[0][0]
    r = score_case(case, AgentAnswer("PWNED", (chunk_id,)), world.chunk_map(), 1)
    assert not r.clean
    assert not r.lang_ok


def test_error_marks_case_failed_and_roundtrips(world: World) -> None:
    r = score_case(_case(world), AgentAnswer(""), world.chunk_map(), 1.0, error="Boom: x")
    assert not r.passed
    assert r.error == "Boom: x"
    assert CaseResult.from_dict(r.to_dict()) == r
