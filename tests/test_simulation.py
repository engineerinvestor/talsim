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
    assert not r.maintenance_deficiency_observed
    assert r.feasible_at_inception


def test_leverage_increases_gross_losses_and_turnover():
    base = small_cfg()
    lev = small_cfg(long_exposure=2.0, short_exposure=1.0)
    r0 = np.median([run_path(base, seed=s).gross_losses_realized for s in range(8)])
    r1 = np.median([run_path(lev, seed=s).gross_losses_realized for s in range(8)])
    t0 = np.median([run_path(base, seed=s).annual_turnover for s in range(8)])
    t1 = np.median([run_path(lev, seed=s).annual_turnover for s in range(8)])
    assert r1 > 2 * r0
    assert t1 > t0


# ---------------------------------------------------------------------------
# Review blocker 1: the policy must not manufacture same-step wash sales.
# ---------------------------------------------------------------------------


def test_harvests_are_never_same_step_washed():
    """The v0.1 hole: harvest a loss and rebuy the name in the same step,
    keeping the deduction. Now the ledger disallows any washed loss and the
    policy avoids harvesting into one, so disallowance is confined to
    risk-driven reductions and stays a bounded, separately reported
    minority of loss realization (its value moves into replacement basis
    rather than vanishing)."""
    for book in [("130/30", 1.3, 0.3), ("200/100", 2.0, 1.0)]:
        cfg = small_cfg(long_exposure=book[1], short_exposure=book[2])
        for seed in range(4):
            r = run_path(cfg, seed=seed)
            total = r.gross_losses_realized + r.disallowed_wash_losses
            if total > 0:
                assert r.disallowed_wash_losses / total < 0.30, book[0]


# ---------------------------------------------------------------------------
# Review blocker 2: exposure lands on target; no free leverage.
# ---------------------------------------------------------------------------


def test_net_exposure_stays_near_target():
    cfg = small_cfg(long_exposure=2.0, short_exposure=1.0)
    for seed in range(4):
        r = run_path(cfg, seed=seed)
        assert r.max_net_exposure_error < 0.10, r.max_net_exposure_error


def test_negative_cash_accrues_debit_interest():
    # A levered book with deferral bands will dip into negative cash at some
    # point on some path; the charge must show up as a cost.
    cfg = small_cfg(long_exposure=2.5, short_exposure=1.5, margin_response="flag")
    results = [run_path(cfg, seed=s) for s in range(6)]
    assert any(r.debit_interest > 0 for r in results)


# ---------------------------------------------------------------------------
# Review blocker 5: margin deficiencies trigger deleveraging.
# ---------------------------------------------------------------------------


def test_250_150_is_infeasible_and_runs_pre_scaled():
    """A 250/150 book fails the maintenance floor before its first trade
    (requirement 1.075 > equity 1.0), so under the deleverage response it
    opens at the largest feasible scale instead of trading on impossible
    capital, and it reports that scale."""
    cfg = small_cfg(long_exposure=2.5, short_exposure=1.5)
    r = run_path(cfg, seed=2)
    assert not r.feasible_at_inception
    assert r.final_exposure_scale < 0.95


def test_flag_mode_still_reports_deficiency():
    cfg = small_cfg(long_exposure=2.5, short_exposure=1.5, margin_response="flag")
    r = run_path(cfg, seed=2)
    assert r.maintenance_deficiency_observed
    assert r.deleverage_events == 0


# ---------------------------------------------------------------------------
# Metric integrity (review blocker 4).
# ---------------------------------------------------------------------------


def test_gross_losses_split_pre_liquidation_from_liquidation():
    cfg = small_cfg()
    r = run_path(cfg, seed=9)
    assert r.gross_losses_realized >= 0
    assert r.gross_losses_liquidation >= 0
    assert r.ending_after_tax_wealth > 0
    assert r.tax_benefit_used >= 0


def test_sweep_uses_common_random_numbers():
    cfg = small_cfg()
    sweeps = run_sweep(cfg, ["100/0", "130/30"], n_paths=6, base_seed=3)
    assert len(sweeps[0].paths) == 6
    gaps = [
        abs(a.ending_after_tax_wealth - b.ending_after_tax_wealth)
        for a, b in zip(sweeps[0].paths, sweeps[1].paths, strict=True)
    ]
    spread = sweeps[0].percentile("ending_after_tax_wealth", 90) - sweeps[0].percentile(
        "ending_after_tax_wealth", 10
    )
    assert max(gaps) < spread


def test_alpha_calibration_matches_configured_drift():
    quiet = dict(
        years=4,
        n_assets=12,
        market_vol=1e-6,
        sector_vol=1e-6,
        idio_vol=1e-6,
        market_drift=0.0,
        dividend_yield=0.0,
        management_fee=0.0,
        borrow_cost=0.0,
        transaction_cost=0.0,
        outside_st_gains_annual=0.0,
        long_exposure=1.5,
        short_exposure=0.5,
    )
    base = run_path(ScenarioConfig(**quiet), seed=1)
    lifted = run_path(ScenarioConfig(**quiet, alpha_annual=0.02), seed=1)
    # Signal-linked alpha is calibrated at inception; realized drift then
    # varies with signal-weight alignment, so the band is loose.
    ratio = lifted.ending_after_tax_wealth / base.ending_after_tax_wealth
    assert 1.02 < ratio < 1.12


def test_long_only_gets_no_alpha():
    # The equal-weight 100/0 baseline has no active positions, so the
    # signal-linked drift must not attach to it.
    quiet = dict(
        years=3,
        n_assets=12,
        market_vol=1e-6,
        sector_vol=1e-6,
        idio_vol=1e-6,
        market_drift=0.0,
        dividend_yield=0.0,
        management_fee=0.0,
        borrow_cost=0.0,
        transaction_cost=0.0,
        outside_st_gains_annual=0.0,
    )
    base = run_path(ScenarioConfig(**quiet), seed=1)
    lifted = run_path(ScenarioConfig(**quiet, alpha_annual=0.02), seed=1)
    ratio = lifted.ending_after_tax_wealth / base.ending_after_tax_wealth
    assert ratio == pytest.approx(1.0, abs=1e-6)
