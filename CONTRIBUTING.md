# Contributing

1. `make install`, then `make lint test gate meta` must all pass.
2. A new metric needs a mutant in `agents.py` that the metric catches, with a test in
   `tests/test_mutants.py`. An eval that has never been seen to fail proves nothing.
3. `baselines/reference.json` changes only in a PR that explains why the reference agent
   legitimately improved. `tests/test_runner_gate_report.py` fails if it drifts silently.
