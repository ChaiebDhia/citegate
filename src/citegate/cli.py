"""Command line interface: run, gate, baseline, meta, compare and bench."""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from citegate import __version__
from citegate.agents import MUTANTS, Agent, ExtractiveAgent
from citegate.corpus import World, build_world
from citegate.gate import GateConfig, baseline_from_results, evaluate_gate, save_baseline
from citegate.llm import PROVIDERS, LLMAgent, build_completer
from citegate.metrics import CaseResult
from citegate.report import html_report, markdown
from citegate.retrieval import BM25Index
from citegate.runner import load_results, run_cases
from citegate.stats import paired_delta
from citegate.summary import summarise

DEFAULTS = GateConfig()


def make_agent(name: str, index: BM25Index, args: argparse.Namespace) -> Agent:
    key = name.removeprefix("mutant:")
    if name == "extractive":
        return ExtractiveAgent(index)
    if key in MUTANTS:
        return MUTANTS[key][0](index)
    if name == "llm":
        try:
            completer = build_completer(
                args.provider,
                args.model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
            )
        except ValueError as err:
            raise SystemExit(f"citegate: {err}") from err
        return LLMAgent(index, completer, name=f"llm:{args.provider}:{args.model}")
    raise SystemExit(f"unknown agent {name!r}")


def _concurrency(args: argparse.Namespace, agent_name: str) -> int:
    if args.concurrency:
        return int(args.concurrency)
    if agent_name != "llm":
        return 32
    return 2 if args.provider == "ollama" else 4


def _setup(args: argparse.Namespace) -> tuple[World, BM25Index]:
    world = build_world(args.services, args.seed)
    index = BM25Index(world.chunks)
    sample = getattr(args, "sample", None)
    return (world.sample(sample) if sample else world), index


def _run(
    args: argparse.Namespace, name: str | None = None
) -> tuple[Agent, World, list[CaseResult]]:
    world, index = _setup(args)
    chosen = name or args.agent
    agent = make_agent(chosen, index, args)
    out = Path(args.out) if getattr(args, "out", None) else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(run_cases(agent, world, concurrency=_concurrency(args, chosen), out=out))
    return agent, world, results


def _write(path: str, text: str) -> None:
    """Write a report, creating missing parent directories first."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _expand(patterns: Sequence[str]) -> list[str]:
    """Expand wildcards ourselves, because PowerShell and cmd.exe do not."""
    files: list[str] = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(pattern)) or [pattern])
    return files


def _emit(text: str, path: str | None) -> None:
    print(text)
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(text + "\n")


def cmd_run(args: argparse.Namespace) -> int:
    agent, _, results = _run(args)
    summary = summarise(agent.name, results)
    _emit(markdown(summary), args.md)
    if args.html:
        _write(args.html, html_report(summary, results))
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    agent, _, results = _run(args)
    save_baseline(Path(args.baseline), agent.name, results)
    print(f"baseline written to {args.baseline} ({len(results)} cases)")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    agent, _, results = _run(args)
    summary = summarise(agent.name, results)
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))["cases"]
    config = GateConfig(args.min_pass_lower_bound, args.min_group_pass, args.tolerance)
    report = evaluate_gate(summary, results, baseline, config)
    _emit(markdown(summary) + "\n" + report.markdown(), args.md)
    if args.html:
        _write(args.html, html_report(summary, results))
    return 0 if report.passed else 1


def cmd_meta(args: argparse.Namespace) -> int:
    """Mutation-test the evals: every deliberately broken agent must be caught."""
    _, _, ref = _run(args, "extractive")
    ref_summary = summarise("extractive", ref)
    baseline = baseline_from_results("extractive", ref)["cases"]
    lines = [
        "| mutant | must be caught by | delta on that metric (95% CI) | gate | verdict |",
        "|---|---|---|---|---|",
    ]
    survivors = 0
    for name, (_, metric) in MUTANTS.items():
        _, _, res = _run(args, name)
        delta, lo, hi = paired_delta(
            [bool(getattr(a, metric)) for a in ref], [bool(getattr(b, metric)) for b in res]
        )
        gate = evaluate_gate(summarise(name, res), res, baseline)  # type: ignore[arg-type]
        caught = hi < 0 and not gate.passed
        survivors += not caught
        lines.append(
            f"| {name} | `{metric}` | {delta:+.1%} ({lo:+.1%} to {hi:+.1%}) | "
            f"{'fail' if not gate.passed else 'pass'} | "
            f"{'caught' if caught else '**SURVIVED**'} |"
        )
    p = ref_summary.groups["ALL"]["passed"]
    lines.append(
        f"\nReference agent: {p.p:.1%} pass. Mutation score: "
        f"{len(MUTANTS) - survivors}/{len(MUTANTS)} defects caught."
    )
    _emit("\n".join(lines), args.md)
    return 0 if survivors == 0 else 1


def cmd_compare(args: argparse.Namespace) -> int:
    """Leaderboard table from one or more JSONL result files (file stem is the run name)."""

    def cell(groups: dict[str, dict[str, object]], group: str) -> str:
        found = groups.get(group)
        return f"{found['passed'].p:.0%}" if found else "n/a"  # type: ignore[attr-defined]

    head = (
        "| run | n | pass (95% CI) | correct | faithful | injection | stale | "
        "out of scope | p50 latency |\n|---|---|---|---|---|---|---|---|---|"
    )
    rows = []
    for path in _expand(args.files):
        if not Path(path).is_file():
            raise SystemExit(f"citegate: no such results file: {path}")
        results = list(load_results(Path(path)).values())
        s = summarise(Path(path).stem, results)
        a = s.groups["ALL"]
        rows.append(
            f"| {Path(path).stem} | {s.n} | {a['passed'].p:.1%} "
            f"({a['passed'].lo:.1%} to {a['passed'].hi:.1%}) | {a['correct'].p:.0%} | "
            f"{a['faithful'].p:.0%} | {cell(s.groups, 'kind:injection')} | "  # type: ignore[arg-type]
            f"{cell(s.groups, 'kind:stale')} | {cell(s.groups, 'kind:out_of_scope')} | "  # type: ignore[arg-type]
            f"{s.latency_p50_ms:,.0f} ms |"
        )
    _emit(head + "\n" + "\n".join(rows) + "\n", args.md)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    world = build_world(args.services, args.seed)
    t1 = time.perf_counter()
    index = BM25Index(world.chunks)
    t2 = time.perf_counter()
    results = asyncio.run(run_cases(ExtractiveAgent(index), world, concurrency=32))
    t3 = time.perf_counter()
    s = summarise("extractive", results)
    print(
        f"cases={len(results)} chunks={len(world.chunks)} build={t1 - t0:.1f}s "
        f"index={t2 - t1:.1f}s run={t3 - t2:.1f}s "
        f"throughput={len(results) / (t3 - t2):,.0f} cases/s "
        f"pass={s.groups['ALL']['passed'].p:.2%} p50={s.latency_p50_ms:.3f}ms "
        f"p95={s.latency_p95_ms:.3f}ms"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="citegate", description=__doc__)
    parser.add_argument("--version", action="version", version=f"citegate {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, agent: bool = True) -> None:
        p.add_argument("--services", type=int, default=48)
        p.add_argument("--seed", type=int, default=7)
        if not agent:
            return
        p.add_argument("--agent", default="extractive", help="extractive, a mutant name, or llm")
        p.add_argument("--sample", type=int, help="stratified subset of N cases (fast, cheap)")
        p.add_argument("--concurrency", type=int, help="default 32, or 2 (ollama) / 4 (hosted)")
        p.add_argument("--provider", choices=sorted(PROVIDERS), default="ollama")
        p.add_argument("--model", default="", help="model id for the llm agent")
        p.add_argument("--base-url", help="override the provider's OpenAI-compatible base URL")
        p.add_argument("--api-key-env", help="environment variable that holds the API key")

    run = sub.add_parser("run", help="run the golden set and print a summary")
    common(run)
    run.add_argument("--out", help="JSONL results file (resumable)")
    run.add_argument("--md", help="append the markdown summary to this file")
    run.add_argument("--html", help="write a single-file HTML report")
    run.set_defaults(func=cmd_run)

    base = sub.add_parser("baseline", help="record a baseline for the regression gate")
    common(base)
    base.add_argument("--baseline", default="baselines/reference.json")
    base.set_defaults(func=cmd_baseline)

    gate = sub.add_parser("gate", help="run and fail (exit 1) on floor or regression")
    common(gate)
    gate.add_argument("--baseline", default="baselines/reference.json")
    gate.add_argument("--min-pass-lower-bound", type=float, default=DEFAULTS.min_pass_lower_bound)
    gate.add_argument("--min-group-pass", type=float, default=DEFAULTS.min_group_pass)
    gate.add_argument("--tolerance", type=float, default=DEFAULTS.tolerance)
    gate.add_argument("--out")
    gate.add_argument("--md", help="append markdown to this file (e.g. $GITHUB_STEP_SUMMARY)")
    gate.add_argument("--html")
    gate.set_defaults(func=cmd_gate)

    meta = sub.add_parser("meta", help="mutation-test the evals themselves")
    common(meta, agent=False)
    meta.add_argument("--md")
    meta.set_defaults(func=cmd_meta_wrapper)

    compare = sub.add_parser("compare", help="leaderboard from JSONL result files")
    compare.add_argument("files", nargs="+")
    compare.add_argument("--md", help="append the table to this file")
    compare.set_defaults(func=cmd_compare)

    bench = sub.add_parser("bench", help="throughput on a scaled-up world")
    common(bench, agent=False)
    bench.set_defaults(func=cmd_bench)
    return parser


def cmd_meta_wrapper(args: argparse.Namespace) -> int:
    for name, default in (("sample", None), ("concurrency", None), ("provider", "ollama")):
        if not hasattr(args, name):
            setattr(args, name, default)
    return cmd_meta(args)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
