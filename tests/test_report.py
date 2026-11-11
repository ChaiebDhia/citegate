from __future__ import annotations

from citegate.metrics import CaseResult
from citegate.report import html_report, markdown
from citegate.summary import summarise


def test_reports_render(reference: list[CaseResult]) -> None:
    summary = summarise("extractive", reference)
    assert "Overall pass **100.0%**" in markdown(summary)
    page = html_report(summary, reference)
    assert page.startswith("<!doctype html>") and "No failing cases" in page


def test_html_report_escapes_failing_answers(reference: list[CaseResult]) -> None:
    sample = list(reference[:5])
    sample[0] = CaseResult.from_dict({**sample[0].to_dict(), "passed": False, "answer": "<x>"})
    assert "&lt;x&gt;" in html_report(summarise("e", sample), sample)
