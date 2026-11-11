from __future__ import annotations

import asyncio
import io
import json
import urllib.error
import urllib.request
from email.message import Message
from typing import Any

import pytest

from citegate.llm import (
    LLMAgent,
    anthropic_completer,
    build_completer,
    openai_compatible_completer,
    post_json,
)
from citegate.retrieval import BM25Index


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["retry-after"] = retry_after
    return urllib.error.HTTPError("http://x", code, "err", headers, io.BytesIO(b""))


def test_llm_agent_parses_citations_and_refusals(index: BM25Index) -> None:
    seen: list[str] = []

    def fake(prompt: str) -> str:
        seen.append(prompt)
        return "REFUSE" if "lunar" in prompt else "The fee is 30 dinars. [s00000-en-fees]"

    agent = LLMAgent(index, fake)
    ok = asyncio.run(agent.answer("How much is the fee for passport renewal (Aldoria)?"))
    assert ok.citations == ("s00000-en-fees",) and "[" not in ok.text and not ok.refused
    assert "ONLY the context" in seen[0]
    assert asyncio.run(agent.answer("How much is the fee for lunar mining permit?")).refused


def test_post_json_retries_transient_errors_then_succeeds() -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def opener(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(429, retry_after="0")
        return io.BytesIO(b'{"ok": 1}')

    assert post_json("http://x", {}, {}, opener=opener, sleep=sleeps.append) == {"ok": 1}
    assert len(calls) == 3 and sleeps == [0.0, 0.0]


def test_post_json_backs_off_exponentially_without_retry_after() -> None:
    sleeps: list[float] = []
    state = {"n": 0}

    def opener(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        state["n"] += 1
        if state["n"] <= 2:
            raise _http_error(503)
        return io.BytesIO(b"{}")

    post_json("http://x", {}, {}, opener=opener, sleep=sleeps.append)
    assert sleeps == [1.0, 2.0]


def test_post_json_does_not_retry_client_errors_and_gives_up() -> None:
    def bad_request(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        raise _http_error(400)

    with pytest.raises(urllib.error.HTTPError):
        post_json("http://x", {}, {}, opener=bad_request, sleep=lambda s: None)

    attempts: list[int] = []

    def always_down(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        attempts.append(1)
        raise _http_error(500)

    with pytest.raises(urllib.error.HTTPError):
        post_json("http://x", {}, {}, retries=2, opener=always_down, sleep=lambda s: None)
    assert len(attempts) == 3


def _capture(reply: dict[str, Any]) -> tuple[list[urllib.request.Request], Any]:
    seen: list[urllib.request.Request] = []

    def opener(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        seen.append(request)
        return io.BytesIO(json.dumps(reply).encode())

    return seen, opener


def test_openai_compatible_completer_local_has_no_auth_header() -> None:
    seen, opener = _capture({"choices": [{"message": {"content": "hi"}}]})
    complete = openai_compatible_completer("gemma3:4b", None, "http://localhost:11434/v1", opener)
    assert complete("q") == "hi"
    assert seen[0].full_url == "http://localhost:11434/v1/chat/completions"
    assert seen[0].get_header("Authorization") is None
    assert json.loads(seen[0].data)["model"] == "gemma3:4b"  # type: ignore[arg-type]


def test_openai_compatible_completer_sends_bearer_key() -> None:
    seen, opener = _capture({"choices": [{"message": {"content": "ok"}}]})
    openai_compatible_completer("m", "sk-test", "https://api.groq.com/openai/v1/", opener)("q")
    assert seen[0].get_header("Authorization") == "Bearer sk-test"
    assert seen[0].full_url == "https://api.groq.com/openai/v1/chat/completions"


def test_anthropic_completer_parses_content_blocks() -> None:
    seen, opener = _capture({"content": [{"type": "text", "text": "hello"}]})
    assert anthropic_completer("some-model", "key", opener)("q") == "hello"
    assert seen[0].get_header("X-api-key") == "key"
    assert seen[0].get_header("Anthropic-version") == "2023-06-01"


def test_build_completer_resolves_presets_and_reports_clear_errors() -> None:
    _, opener = _capture({"choices": [{"message": {"content": "x"}}]})
    assert callable(build_completer("ollama", "gemma3:4b", environ={}, opener=opener))
    assert callable(build_completer("groq", "m", environ={"GROQ_API_KEY": "k"}, opener=opener))
    assert callable(build_completer("anthropic", "m", environ={"ANTHROPIC_API_KEY": "k"}))
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        build_completer("groq", "m", environ={})
    with pytest.raises(ValueError, match="unknown provider"):
        build_completer("nope", "m", environ={})
    with pytest.raises(ValueError, match="--model"):
        build_completer("ollama", "", environ={})
    with pytest.raises(ValueError, match="--base-url"):
        build_completer("openai-compatible", "m", environ={"OPENAI_API_KEY": "k"})
    custom = build_completer(
        "openai-compatible",
        "m",
        base_url="http://vllm:8000/v1",
        api_key_env="MY_KEY",
        environ={"MY_KEY": "k"},
        opener=opener,
    )
    assert callable(custom)
