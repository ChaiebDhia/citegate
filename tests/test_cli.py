from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from citegate.cli import main


def test_cli_run_writes_html_jsonl_and_markdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    html, out, md = tmp_path / "r.html", tmp_path / "sub" / "r.jsonl", tmp_path / "s.md"
    code = main(["run", "--html", str(html), "--out", str(out), "--md", str(md)])
    assert code == 0
    assert html.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert len(out.read_text(encoding="utf-8").splitlines()) == 744
    assert "Overall pass" in md.read_text(encoding="utf-8")
    assert "Overall pass" in capsys.readouterr().out


def test_cli_gate_exit_codes(tmp_path: Path) -> None:
    baseline = tmp_path / "b.json"
    assert main(["baseline", "--baseline", str(baseline)]) == 0
    assert main(["gate", "--baseline", str(baseline), "--md", str(tmp_path / "g.md")]) == 0
    assert main(["gate", "--agent", "mutant:hallucinator", "--baseline", str(baseline)]) == 1


def test_cli_meta_and_bench(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["meta"]) == 0
    assert "7/7 defects caught" in capsys.readouterr().out
    assert main(["bench", "--services", "16"]) == 0
    assert "throughput=" in capsys.readouterr().out


def test_cli_rejects_unknown_agent() -> None:
    with pytest.raises(SystemExit):
        main(["run", "--agent", "does-not-exist"])


def test_cli_llm_agent_with_fake_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("citegate.cli.build_completer", lambda *a, **k: lambda prompt: "REFUSE")
    out = tmp_path / "llm.jsonl"
    args = ["run", "--agent", "llm", "--provider", "ollama", "--model", "fake", "--sample", "24"]
    assert main([*args, "--out", str(out)]) == 0
    assert "llm:ollama:fake" in capsys.readouterr().out
    assert len(out.read_text(encoding="utf-8").splitlines()) == 24
    assert main([*args, "--out", str(out)]) == 0  # resumed, nothing re-run
    assert len(out.read_text(encoding="utf-8").splitlines()) == 24


def test_cli_llm_agent_missing_key_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="GROQ_API_KEY"):
        main(["run", "--agent", "llm", "--provider", "groq", "--model", "x", "--sample", "3"])


def test_cli_compare_builds_a_leaderboard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good, bad, tiny = tmp_path / "good.jsonl", tmp_path / "bad.jsonl", tmp_path / "tiny.jsonl"
    main(["run", "--sample", "24", "--out", str(good)])
    main(["run", "--agent", "hallucinator", "--sample", "24", "--out", str(bad)])
    main(["run", "--sample", "3", "--out", str(tiny)])
    capsys.readouterr()
    md = tmp_path / "board.md"
    assert main(["compare", str(good), str(bad), str(tiny), "--md", str(md)]) == 0
    text = capsys.readouterr().out
    assert "| good | 24 |" in text and "| bad | 24 |" in text and "n/a" in text
    assert md.read_text(encoding="utf-8").startswith("| run |")


def test_module_entry_point_reports_version() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "citegate", "--version"], capture_output=True, text=True, check=True
    )
    assert proc.stdout.startswith("citegate ")


def test_cli_creates_missing_output_directories(tmp_path: Path) -> None:
    html = tmp_path / "docs" / "results" / "r.html"
    md = tmp_path / "docs" / "results" / "r.md"
    jsonl = tmp_path / "runs" / "nested" / "r.jsonl"
    assert main(["run", "--sample", "24", "--html", str(html), "--md", str(md)]) == 0
    assert html.exists() and md.exists()
    main(["run", "--sample", "24", "--out", str(jsonl)])
    board = tmp_path / "docs" / "leaderboard" / "board.md"
    assert main(["compare", str(jsonl), "--md", str(board)]) == 0
    assert board.exists()
