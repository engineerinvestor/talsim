"""target_weights must reproduce the original scalar grid search bitwise."""

import numpy as np
import pytest

from talsim import BOOK_PRESETS
from talsim.optimize import target_weights


def _reference_target_weights(signal, long_exposure, short_exposure):
    """The pre-0.4.1 implementation: a scalar loop over the scale grid."""
    k = len(signal)
    base = np.full(k, (long_exposure - short_exposure) / k)
    if short_exposure == 0:
        return np.full(k, long_exposure / k)
    ranks = np.argsort(np.argsort(signal))
    tilt = ranks - ranks.mean()
    scale_grid = np.linspace(0.1, 50.0, 500)
    best = base
    best_err = np.inf
    for scale in scale_grid:
        w = base + tilt / np.abs(tilt).sum() * scale
        short_sum = -w[w < 0].sum()
        err = abs(short_sum - short_exposure)
        if err < best_err:
            best_err = err
            best = w
    w = best.copy()
    neg = w < 0
    if w[neg].sum() != 0:
        w[neg] *= short_exposure / (-w[neg].sum())
    pos = ~neg
    w[pos] *= long_exposure / w[pos].sum()
    return w


BOOKS = list(BOOK_PRESETS.values()) + [(2.334, 1.334)]  # plus the net-preserving 250/150 case


@pytest.mark.parametrize("k", [12, 36, 100])
def test_vectorized_grid_search_matches_reference_bitwise(k):
    rng = np.random.default_rng(k)
    for long_exposure, short_exposure in BOOKS:
        for _ in range(100):
            signal = rng.standard_normal(k)
            expected = _reference_target_weights(signal, long_exposure, short_exposure)
            actual = target_weights(signal, long_exposure, short_exposure)
            assert np.array_equal(actual, expected)


def test_long_only_is_equal_weight():
    w = target_weights(np.random.default_rng(1).standard_normal(20), 1.0, 0.0)
    assert np.array_equal(w, np.full(20, 0.05))


def test_gross_targets_are_exact():
    w = target_weights(np.random.default_rng(2).standard_normal(36), 1.5, 0.5)
    assert w[w > 0].sum() == pytest.approx(1.5)
    assert -w[w < 0].sum() == pytest.approx(0.5)
