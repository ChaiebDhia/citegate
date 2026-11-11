from __future__ import annotations

import pytest

from citegate.stats import paired_delta, percentile, wilson


def test_wilson_known_values() -> None:
    p = wilson(50, 100)
    assert p.lo == pytest.approx(0.4038, abs=1e-3)
    assert p.hi == pytest.approx(0.5962, abs=1e-3)
    assert wilson(0, 10).lo == pytest.approx(0.0, abs=1e-12)
    assert wilson(10, 10).hi == pytest.approx(1.0, abs=1e-12)
    assert wilson(0, 0).hi == 1.0


def test_paired_delta_no_change_and_regression() -> None:
    same = [True, False, True, True]
    assert paired_delta(same, same) == (0.0, 0.0, 0.0)
    base = [True] * 200
    cand = [True] * 180 + [False] * 20
    delta, lo, hi = paired_delta(base, cand)
    assert delta == pytest.approx(-0.10)
    assert hi < 0
    assert lo < delta


def test_paired_delta_validates_input() -> None:
    with pytest.raises(ValueError, match="equal length"):
        paired_delta([True], [True, False])
    assert paired_delta([], []) == (0.0, 0.0, 0.0)


def test_percentile() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([1, 2, 3, 4], 100) == 4
