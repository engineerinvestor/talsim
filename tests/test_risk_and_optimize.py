import numpy as np
import pytest

from talsim.config import ScenarioConfig
from talsim.optimize import target_weights
from talsim.risk import (
    ewma_cov,
    ledoit_wolf_cov,
    nearest_psd,
    oas_cov,
    sample_cov,
)


def _returns(t=120, n=10, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((t, n)) * 0.02


def test_estimators_return_symmetric_psd():
    x = _returns()
    for estimator in (sample_cov, ewma_cov, ledoit_wolf_cov, oas_cov):
        cov = estimator(x)
        assert cov.shape == (10, 10)
        assert np.allclose(cov, cov.T)
        assert np.linalg.eigvalsh(cov).min() > -1e-10


def test_shrinkage_reduces_condition_number():
    # With T barely above N, shrinkage should tame the condition number.
    x = _returns(t=12, n=10)
    raw = np.linalg.cond(sample_cov(x))
    shrunk = np.linalg.cond(ledoit_wolf_cov(x))
    assert shrunk < raw


def test_nearest_psd_repairs_indefinite_matrix():
    bad = np.array([[1.0, 0.99, 0.0], [0.99, 1.0, -0.99], [0.0, -0.99, 1.0]])
    bad[0, 2] = 0.99  # deliberately inconsistent correlations
    bad[2, 0] = 0.99
    fixed = nearest_psd(bad)
    assert np.linalg.eigvalsh(fixed).min() >= 0


def test_target_weights_hit_exposures_long_short():
    rng = np.random.default_rng(3)
    signal = rng.standard_normal(36)
    w = target_weights(signal, 1.5, 0.5)
    assert w[w > 0].sum() == pytest.approx(1.5, abs=1e-6)
    assert -w[w < 0].sum() == pytest.approx(0.5, abs=1e-6)
    # Strongest signals long, weakest short.
    assert w[np.argmax(signal)] > 0
    assert w[np.argmin(signal)] < 0


def test_target_weights_long_only_is_equal_weight():
    # The 100/0 baseline is a passive equal-weight harvesting portfolio;
    # giving it a signal tilt would contaminate the leverage comparison.
    rng = np.random.default_rng(4)
    signal = rng.standard_normal(36)
    w = target_weights(signal, 1.0, 0.0)
    assert np.allclose(w, 1.0 / 36)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_config_presets_and_validation():
    cfg = ScenarioConfig().with_book("130/30")
    assert cfg.gross_exposure == pytest.approx(1.6)
    assert cfg.active_gross == pytest.approx(0.6)
    with pytest.raises(ValueError):
        ScenarioConfig(long_exposure=0.5, short_exposure=0.5)
