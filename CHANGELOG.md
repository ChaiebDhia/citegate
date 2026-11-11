# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-11

### Added
- Deterministic, trilingual (English, French, Arabic) synthetic public-service corpus with 744
  golden cases covering lookups, multi-hop questions, out-of-scope questions, planted prompt
  injection, stale guidance, forged citations and language drift.
- Reference extractive agent and seven deliberately broken "mutant" agents that mutation-test the
  evals themselves (`citegate meta`, 7 of 7 defects caught).
- Eight deterministic metrics: `correct`, `faithful`, `citations_valid`, `cites_gold`,
  `refusal_ok`, `clean`, `lang_ok` and the combined `passed`.
- Statistical regression gate: Wilson lower-bound floor plus paired (McNemar-style) deltas against
  a committed baseline (`citegate gate`, exit code 1 on failure).
- Async, resumable runner with bounded concurrency that records agent crashes as failed cases.
- Provider adapters for any OpenAI-compatible endpoint (Ollama, Groq, OpenAI, vLLM) and
  Anthropic, using only the standard library, with retry and exponential backoff on 429 and 5xx.
- `citegate compare` leaderboard, `--sample` stratified subsets, Markdown and single-file HTML
  reports, and a `bench` command.
- GitHub Actions: lint and types, test matrix (Python 3.11 to 3.13), eval gate with job summary,
  Docker smoke test, CodeQL, live report on GitHub Pages and tagged releases.
