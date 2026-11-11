# ADR 3: gate on intervals, not on point estimates

**Status** accepted

**Context** A fixed threshold on a noisy score either flakes or hides real regressions.

**Decision** The gate requires the Wilson 95 % lower bound of the pass rate to clear a floor, and
compares each metric to a stored baseline with a paired (McNemar-style) interval. It fails on a
significant drop or on any drop larger than the tolerance.

**Consequences** Deterministic agents get exact deltas. Stochastic LLM agents get a gate that
does not flip on one unlucky sample. The normal approximation is weak below roughly 30
discordant pairs, so use more cases for small effects.
