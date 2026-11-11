# ADR 1: deterministic metrics before any LLM judge

**Status** accepted

**Context** CI needs verdicts that are free, reproducible and runnable without secrets. An LLM
judge is none of those, and its own errors are hard to separate from the agent's.

**Decision** Faithfulness, citation integrity, refusal, injection hygiene, freshness and language
fidelity are scored by deterministic checks over a fictional corpus, so pre-training knowledge
cannot mask a grounding failure. LLM agents plug in through the same `Agent` interface.

**Consequences** The checks are proxies. Token support and number matching miss subtle semantic
errors, so they are a regression floor, not a replacement for human review of live traffic.
