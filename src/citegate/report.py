"""Markdown (for CI job summaries) and a single-file HTML report (for humans)."""

from __future__ import annotations

import html
from collections.abc import Sequence

from citegate.metrics import METRICS, CaseResult
from citegate.summary import Summary


def markdown(summary: Summary) -> str:
    head = "| group | n | " + " | ".join(METRICS) + " |\n|---|---|" + "---|" * len(METRICS)
    rows = []
    for name, metrics in summary.groups.items():
        cells = " | ".join(f"{metrics[m].p:.1%}" for m in METRICS)
        rows.append(f"| {name} | {metrics['passed'].n} | {cells} |")
    overall = summary.groups["ALL"]["passed"]
    return (
        f"### citegate results for `{summary.agent}`\n\n"
        f"Overall pass **{overall.p:.1%}** (95% CI {overall.lo:.1%} to {overall.hi:.1%}, "
        f"n={summary.n}). Latency p50 {summary.latency_p50_ms:.2f} ms, "
        f"p95 {summary.latency_p95_ms:.2f} ms.\n\n{head}\n" + "\n".join(rows) + "\n"
    )


_CSS = """
:root{--bg:#fff;--fg:#14181f;--mut:#5b6573;--line:#dfe3e8;--ok:#1a7f4b;--bad:#c2332b;--bar:#2f6fed}
@media(prefers-color-scheme:dark){:root{--bg:#0e1116;--fg:#e6e9ee;--mut:#9aa4b2;--line:#262c36;
--ok:#4cc38a;--bad:#f2766d;--bar:#6e9bff}}
body{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0 auto;
max-width:980px;padding:24px}h1{font-size:22px}h2{font-size:16px;margin-top:28px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--line);
padding:6px 8px;text-align:left}th{color:var(--mut);font-weight:600}.num{text-align:right;
font-variant-numeric:tabular-nums}.cards{display:flex;gap:12px;flex-wrap:wrap}
.card{border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:150px}
.card b{display:block;font-size:22px}.bad{color:var(--bad)}.ok{color:var(--ok)}
code{font-size:12px}.wrap{overflow-x:auto}
"""


def _bar(p: float, lo: float, hi: float, width: int = 180) -> str:
    def x(v: float) -> float:
        return round(v * width, 1)

    return (
        f'<svg width="{width}" height="14" role="img" aria-label="{p:.1%}">'
        f'<rect width="{x(p)}" height="14" fill="var(--bar)" opacity=".35"/>'
        f'<line x1="{x(lo)}" x2="{x(hi)}" y1="7" y2="7" stroke="var(--fg)" stroke-width="2"/></svg>'
    )


def html_report(summary: Summary, results: Sequence[CaseResult]) -> str:
    overall = summary.groups["ALL"]
    cards = "".join(
        f'<div class="card">{html.escape(label)}<b>{overall[m].p:.1%}</b>'
        f'<span style="color:var(--mut)">CI {overall[m].lo:.1%} to {overall[m].hi:.1%}</span></div>'
        for label, m in (
            ("Pass", "passed"),
            ("Correct", "correct"),
            ("Faithful", "faithful"),
            ("Cites gold", "cites_gold"),
            ("Clean", "clean"),
        )
    )
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td class='num'>{m['passed'].n}</td>"
        f"<td>{_bar(m['passed'].p, m['passed'].lo, m['passed'].hi)}</td>"
        + "".join(f"<td class='num'>{m[k].p:.1%}</td>" for k in METRICS)
        + "</tr>"
        for name, m in summary.groups.items()
    )
    head = "".join(f"<th>{k}</th>" for k in METRICS)
    failing = [r for r in results if not r.passed][:25]
    fails = (
        "".join(
            f"<tr><td><code>{html.escape(r.case_id)}</code></td><td>{html.escape(r.question)}</td>"
            f"<td>{html.escape(r.answer[:140] or (r.error or ''))}</td></tr>"
            for r in failing
        )
        or '<tr><td colspan="3" class="ok">No failing cases.</td></tr>'
    )
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>citegate report: {html.escape(summary.agent)}</title><style>{_CSS}</style>"
        f"<h1>citegate report <code>{html.escape(summary.agent)}</code></h1>"
        f"<p>{summary.n} cases. Latency p50 {summary.latency_p50_ms:.2f} ms, "
        f"p95 {summary.latency_p95_ms:.2f} ms. Bars show the pass rate with its 95% Wilson "
        f"interval.</p><div class='cards'>{cards}</div><h2>Breakdown</h2>"
        f"<div class='wrap'><table><tr><th>group</th><th>n</th><th>pass (95% CI)</th>{head}</tr>"
        f"{rows}</table></div><h2>First failing cases</h2><div class='wrap'><table>"
        f"<tr><th>case</th><th>question</th><th>answer</th></tr>{fails}</table></div></html>"
    )
