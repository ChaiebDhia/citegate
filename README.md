# citegate

**Evals as unit tests for citation-grounded public-sector AI agents.** Multilingual (English, French, Arabic), statistically gated, and mutation-tested. Zero runtime dependencies.

[![CI](https://github.com/ChaiebDhia/citegate/actions/workflows/ci.yml/badge.svg)](https://github.com/ChaiebDhia/citegate/actions/workflows/ci.yml)
[![CodeQL](https://github.com/ChaiebDhia/citegate/actions/workflows/codeql.yml/badge.svg)](https://github.com/ChaiebDhia/citegate/actions/workflows/codeql.yml)
[![Release](https://img.shields.io/github/v/release/ChaiebDhia/citegate)](https://github.com/ChaiebDhia/citegate/releases)
[![Live report](https://img.shields.io/badge/live%20report-GitHub%20Pages-2f6fed)](https://chaiebdhia.github.io/citegate/)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![types](https://img.shields.io/badge/mypy-strict-informational)
![license](https://img.shields.io/badge/license-MIT-green)

**Live demo:** [chaiebdhia.github.io/citegate](https://chaiebdhia.github.io/citegate/) is rebuilt by CI on every push to `main`.

A government chatbot that invents a fee, cites a document that does not exist, obeys an instruction hidden in a web page, or quotes last year's policy is a production incident. `citegate` turns those failure modes into a test suite that runs on every pull request and **fails the build when reliability regresses**.

## Why it is different

1. **The evals are tested too.** Seven deliberately broken agents ("mutants") ship in the repo, each breaking one property. CI fails if any mutant slips past the gate. An eval that has never been seen to fail proves nothing.
2. **Gates on evidence, not noise.** The gate needs the Wilson 95 % lower bound of the pass rate to clear a floor and compares every metric with a stored baseline using a paired (McNemar-style) interval, so it does not flake on a lucky sample and does not hide real regressions.
3. **Fictional world, so grounding is provable.** The corpus is invented. A correct answer cannot come from pre-training, it must come from the cited passage.
4. **Public-sector failure modes built in.** Out-of-scope questions, prompt injection planted in a document, stale (archived) guidance, forged citations, and language drift for French and Arabic users.
5. **Hermetic CI.** The reference agent is extractive and deterministic, so the pipeline needs no API key. LLM agents plug into the same interface.

```mermaid
flowchart LR
  W["Synthetic world<br/>EN / FR / AR"] --> R["Async runner<br/>resumable JSONL"]
  A["Agent under test"] --> R
  MUT["7 mutant agents"] -.-> R
  R --> M["Deterministic metrics"]
  M --> S["Wilson CIs and<br/>paired deltas"]
  S --> G{"Gate"}
  G -->|pass| OK["merge"]
  G -->|fail| X["block PR<br/>HTML report"]
```

## Quickstart

```bash
pip install -e ".[dev]"
citegate run --html report.html          # score the reference agent, write a report
citegate gate                            # exit 1 on floor or regression failure
citegate meta                            # mutation-test the evals themselves
citegate bench --services 2000           # throughput on a 30k-case world
pytest                                   # 64 tests, 99 % coverage
```

## Results (reference agent, 744 golden cases, seed 7)

| group | n | pass |
|---|---|---|
| overall | 744 | 100.0 % (95 % CI 99.5 to 100.0) |
| fee, days, age, office, multihop | 672 | 100 % |
| injection, stale, out of scope | 72 | 100 % |
| en / fr / ar | 248 each | 100 % |

**Mutation score 7 / 7.** Every seeded defect is caught by the metric designed for it, and the gate fails:

| mutant | caught by | paired delta (95 % CI) |
|---|---|---|
| `stale_picker` | `clean` | -3.2 % (-4.5 to -2.0) |
| `never_refuser` | `refusal_ok` | -3.2 % (-4.5 to -2.0) |
| `over_refuser` | `correct` | -96.8 % (-98.0 to -95.5) |
| `hallucinator` | `faithful` | -75.8 % (-78.9 to -72.7) |
| `citation_forger` | `citations_valid` | -96.8 % (-98.0 to -95.5) |
| `naive_copy` | `clean` | -3.2 % (-4.5 to -2.0) |
| `language_drift` | `lang_ok` | -64.5 % (-68.0 to -61.1) |

**Throughput.** `citegate bench --services 2000` scores 30,024 cases over 14,001 chunks in 47 s on one core (638 cases/s, p95 2.7 ms per case) with the reference agent. Numbers depend on your machine, run it yourself.

## Metrics

| metric | meaning |
|---|---|
| `correct` | all gold values present, or a refusal when one is required |
| `faithful` | cited, valid ids, every number supported by the cited text, at least 70 % of content tokens supported |
| `citations_valid` | every cited id exists in the corpus (catches forged citations) |
| `cites_gold` | each required fact is backed by an acceptable chunk |
| `refusal_ok` | refuses exactly when the question is out of scope |
| `clean` | no forbidden string leaked (injection canary, superseded value) |
| `lang_ok` | the answer language equals the question language |

## Evaluate a real model for free

The same golden set runs against any model that speaks the OpenAI `/chat/completions` protocol, plus Anthropic. Runs stream to JSONL and **resume** after a crash or rate limit, and a crashing agent call is recorded as a failed case, never as a crashed run. Transient `429` and `5xx` replies are retried with exponential backoff.

**Local, no key, no quota (recommended).** Install [Ollama](https://ollama.com), then:

```bash
ollama pull gemma3:4b
citegate run --agent llm --provider ollama --model gemma3:4b \
  --sample 150 --out runs/gemma3-4b.jsonl --html runs/gemma3-4b.html
```

**Hosted free tier.** A free [Groq](https://console.groq.com) key is enough for a few hundred cases (check the console for current limits):

```bash
export GROQ_API_KEY=...        # PowerShell: $env:GROQ_API_KEY="..."
citegate run --agent llm --provider groq --model llama-3.1-8b-instant \
  --sample 300 --out runs/llama-3.1-8b.jsonl
```

`--sample N` takes a deterministic stratified subset, so every failure mode and language stays represented. Compare runs in one table:

```bash
citegate compare runs/*.jsonl --md docs/results/leaderboard.md
```

Other endpoints: `--provider openai-compatible --base-url http://host:8000/v1 --api-key-env MY_KEY` (vLLM, LM Studio, OpenAI), or `--provider anthropic`. To evaluate your own agent, implement one method:

```python
class MyAgent:
    name = "my-agent"

    async def answer(self, question: str) -> AgentAnswer: ...
```

Live provider calls are not exercised in CI because they need a model or credentials. The adapters, retry logic and provider resolution are covered by tests against a fake transport.

## Design notes

* [ADR 1](docs/adr/0001-deterministic-metrics-first.md) deterministic metrics before any LLM judge
* [ADR 2](docs/adr/0002-mutation-testing-the-evals.md) mutation-test the evals
* [ADR 3](docs/adr/0003-statistical-gate.md) gate on intervals, not point estimates

## Limitations

* The golden set is synthetic and template-generated. It measures regression risk, not real-world accuracy. Use it as a floor next to human review of live traffic.
* Token and number support are proxies for faithfulness. They miss subtle semantic errors, which is why an LLM-judge adapter is on the roadmap rather than hidden inside the default gate.
* The reference agent's injection filter is a deliberately simple heuristic. It is a baseline, not a defence.
* The French and Arabic templates are short and regular. A native speaker should extend them with real administrative phrasing before the set is used to judge a production system.
* The paired interval uses a normal approximation and is weak below roughly 30 discordant pairs.

## Roadmap

LLM-judge adapter with calibration against human labels, a TypeScript/React viewer for run history, more languages, retrieval-quality metrics (recall at k) reported separately from generation quality.

## License

MIT
