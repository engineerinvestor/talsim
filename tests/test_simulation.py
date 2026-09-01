import numpy as np
import pytest

from talsim.config import ScenarioConfig
from talsim.simulation import run_path, run_sweep


def small_cfg(**overrides):
    defaults = dict(years=3, n_assets=12, outside_st_gains_annual=50_000.0)
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def test_run_path_is_deterministic_for_a_seed():
    cfg = small_cfg()
    a = run_path(cfg, seed=11)
    b = run_path(cfg, seed=11)
    assert a.ending_after_tax_wealth == pytest.approx(b.ending_after_tax_wealth)
    assert a.gross_losses_realized == pytest.approx(b.gross_losses_realized)


def test_long_only_never_pays_borrow_or_pil():
    cfg = small_cfg()
    r = run_path(cfg, seed=5)
    assert r.borrow_costs == 0.0
    assert r.payments_in_lieu == 0.0
    assert not r.margin_breached


def test_leverage_increases_gross_losses_and_turnover():
    base = small_cfg()
    lev = small_cfg(long_exposure=2.0, short_exposure=1.0)
    r0 = np.median([run_path(base, seed=s).gross_losses_realized for s in range(8)])
    r1 = np.median([run_path(lev, seed=s).gross_losses_realized for s in range(8)])
    t0 = np.median([run_path(base, seed=s).annual_turnover for s in range(8)])
    t1 = np.median([run_path(lev, seed=s).annual_turnover for s in range(8)])
    assert r1 > 2 * r0
    assert t1 > t0


def test_250_150_breaches_maintenance_margin_from_the_start():
    cfg = small_cfg(long_exposure=2.5, short_exposure=1.5)
    r = run_path(cfg, seed=2)
    # Equity 1.0 vs requirement 0.25*2.5 + 0.30*1.5 = 1.075 > 1.0.
    assert r.margin_breached
    assert r.min_margin_excess_ratio < 0


def test_wealth_accounting_is_internally_consistent():
    cfg = small_cfg()
    r = run_path(cfg, seed=9)
    assert r.ending_after_tax_wealth > 0
    assert r.tax_benefit_used >= 0
    assert r.gross_losses_realized >= r.net_loss_pre_liquidation


def test_sweep_uses_common_random_numbers():
    cfg = small_cfg()
    sweeps = run_sweep(cfg, ["100/0", "130/30"], n_paths=4, base_seed=3)
    assert len(sweeps[0].paths) == 4
    # CRN: the same seed drives both books, so market luck is shared and the
    # cross-book wealth gap is far tighter than the cross-path spread.
    gaps = [
        abs(a.ending_after_tax_wealth - b.ending_after_tax_wealth)
        for a, b in zip(sweeps[0].paths, sweeps[1].paths, strict=True)
    ]
    spread = sweeps[0].percentile("ending_after_tax_wealth", 90) - sweeps[0].percentile(
        "ending_after_tax_wealth", 10
    )
    assert max(gaps) < spread
