# ADR 2: mutation-test the evals

**Status** accepted

**Context** Most eval suites are never checked for false negatives. A metric that cannot fail is
decoration.

**Decision** `agents.py` ships seven deliberately broken agents, each breaking one property. CI
runs `citegate meta` and fails if any mutant passes the gate or if the metric designed to catch it
does not show a statistically significant drop.

**Consequences** Adding a metric means adding the defect it is meant to detect.
