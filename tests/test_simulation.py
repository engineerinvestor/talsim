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


def test_250_150_is_infeasible_and_runs_net_preserving_extension():
    """A 250/150 book fails the maintenance floor before its first trade
    (requirement 1.075 > equity 1.0). The feasibility scaling keeps the
    100% net core and shrinks only the long/short extension: with the 2%
    buffer, e_max = (0.98 - 0.25) / 0.55 = 1.327, i.e. roughly 233/133."""
    cfg = small_cfg(long_exposure=2.5, short_exposure=1.5)
    r = run_path(cfg, seed=2)
    assert not r.feasible_at_inception
    assert r.extension_scale == pytest.approx(1.327 / 1.5, abs=0.01)
    # Net exposure stays near 1.0 despite the scaling.
    assert abs((r.avg_long_exposure - r.avg_short_exposure) - 1.0) < 0.15


def test_insolvency_terminates_and_settles_the_actual_year():
    """Find a genuinely insolvent path (extreme gross exposure in flag mode
    with violent idiosyncratic moves) and assert the branch really ran:
    early termination, and the terminal year's realized losses actually
    generating benefit against that year's outside gains instead of being
    settled against the far-future horizon year."""
    cfg = small_cfg(
        long_exposure=4.5,
        short_exposure=3.5,
        idio_vol=0.5,
        margin_response="flag",
        outside_st_gains_annual=50_000.0,
    )
    hit = None
    for seed in range(25):
        r = run_path(cfg, seed=seed)
        if r.insolvent:
            hit = r
            break
    assert hit is not None, "no insolvent path found; test setup too tame"
    assert hit.termination_step < cfg.n_steps - 1
    assert hit.gross_losses_realized > 0
    # The terminal year settles with its own outside gains, so realized
    # losses buy a material benefit (the pre-fix code reported ~$0).
    assert hit.tax_benefit_used > 1_000
    import math

    for value in vars(hit).values():
        if isinstance(value, float):
            assert math.isfinite(value)


def test_infeasible_net_core_is_rejected_in_deleverage_mode():
    """A 500/0 book needs 125% of NAV in maintenance with no extension to
    shrink; the validator must refuse it rather than let the margin
    response silently fail."""
    with pytest.raises(ValueError):
        ScenarioConfig(long_exposure=5.0, short_exposure=0.0)
    # The same book is representable in flag mode, which never claims to cure.
    cfg = ScenarioConfig(long_exposure=5.0, short_exposure=0.0, margin_response="flag")
    assert cfg.gross_exposure == pytest.approx(5.0)


def test_config_validation_rejects_nonsense():
    for bad in (
        dict(n_sectors=0),
        dict(alpha_reference_active_gross=0),
        dict(management_fee=-0.01),
        dict(signal_autocorr=1.1),
        dict(long_maintenance=-0.1),
        dict(deleverage_buffer=0.0),
        dict(harvest_exposure_floor=1.5),
        dict(management_fee=float("nan")),
        dict(outside_st_gains_annual=float("inf")),
        dict(outside_st_gains_annual=-1.0),
        dict(ordinary_offset_limit=-3000.0),
        dict(outside_st_gain_events={0: float("nan")}),
        dict(outside_st_gain_events={0: float("inf")}),
        dict(outside_st_gain_events={0: -1.0}),
        dict(outside_st_gain_events={-1: 5.0}),
        dict(outside_st_gain_events={10: 5.0}),
        dict(outside_st_gain_events={"a": 5.0}),
        dict(outside_st_gain_events={True: 5.0}),
        dict(outside_st_gain_events={0: "5"}),
    ):
        with pytest.raises(ValueError):
            ScenarioConfig(**bad)
    # A valid event schedule still constructs.
    cfg = ScenarioConfig(outside_st_gain_events={2: 500_000.0})
    assert cfg.outside_st_gain_for_year(2) == pytest.approx(600_000.0)


def test_flag_mode_still_reports_deficiency():
    cfg = small_cfg(long_exposure=2.5, short_exposure=1.5, margin_response="flag")
    r = run_path(cfg, seed=2)
    assert r.maintenance_deficiency_observed
    assert r.deleverage_events == 0


# ---------------------------------------------------------------------------
# Metric integrity (review blocker 4).
# ---------------------------------------------------------------------------


def _spy_on_closes(monkeypatch):
    """Record every ledger close (step and realized records) during a run."""
    from talsim.lots import Ledger

    calls: list[tuple[int, list]] = []
    original = Ledger.close

    def spy(self, side, asset, shares, price, step, prefer_gains=False):
        out = original(self, side, asset, shares, price, step, prefer_gains)
        calls.append((step, out))
        return out

    monkeypatch.setattr(Ledger, "close", spy)
    return calls


def test_gross_losses_split_pre_liquidation_from_liquidation(monkeypatch):
    calls = _spy_on_closes(monkeypatch)
    cfg = small_cfg()
    r = run_path(cfg, seed=9)
    assert r.gross_losses_realized >= 0
    assert r.gross_losses_liquidation >= 0
    assert r.ending_after_tax_wealth > 0
    assert r.tax_benefit_used >= 0
    # The unwind is the tail of the close sequence at the terminal step:
    # one close per (side, asset) still held, each taking the whole position.
    last_step = cfg.n_steps - 1
    tail = [recs for step, recs in calls if step == last_step]
    n_liq = sum(1 for step, _ in calls if step == last_step) - sum(
        1 for step, recs in calls if step == last_step and not recs
    )
    assert n_liq > 0
    liq_recs = [rec for recs in tail for rec in recs]
    # Everything closed at the terminal step (rebalance and unwind) is
    # either pre-liquidation or liquidation, never both.
    all_recs = [rec for _, recs in calls for rec in recs]
    total_losses = sum(-rec.gain for rec in all_recs if rec.gain < 0)
    assert r.gross_losses_realized + r.gross_losses_liquidation == pytest.approx(total_losses)
    assert r.gross_losses_liquidation <= sum(-rec.gain for rec in liq_recs if rec.gain < 0) + 1e-6


def test_liquidation_occurs_at_terminal_step(monkeypatch):
    """Inception lots open at step -1 and the loop ends at n_steps - 1, so
    the unwind must share that step. On an exactly-one-year horizon the
    inception lots have been held exactly 365 days: short term, per Pub
    550's "more than one year". Before the fix the unwind was stamped one
    step later and every such lot was long term after 456 days."""
    calls = _spy_on_closes(monkeypatch)

    def inception_longs_closed_at(step):
        return [
            rec
            for s, recs in calls
            if s == step
            for rec in recs
            if rec.side == "long" and rec.open_step == -1
        ]

    cfg = small_cfg(years=1, steps_per_year=4, harvest_threshold=10.0)
    r = run_path(cfg, seed=3)
    assert not r.insolvent
    assert max(step for step, _ in calls) == cfg.n_steps - 1 == 3
    liq_longs = inception_longs_closed_at(3)
    assert liq_longs, "no inception long lots survived to liquidation"
    assert {rec.term for rec in liq_longs} == {"st"}

    # One more year and the same lots are long term (730 days > 365).
    calls.clear()
    cfg2 = small_cfg(years=2, steps_per_year=4, harvest_threshold=10.0)
    run_path(cfg2, seed=3)
    assert max(step for step, _ in calls) == 7
    liq_longs = inception_longs_closed_at(7)
    assert liq_longs and {rec.term for rec in liq_longs} == {"lt"}


def test_insolvent_path_liquidates_at_termination_step(monkeypatch):
    calls = _spy_on_closes(monkeypatch)
    cfg = small_cfg(
        long_exposure=4.5,
        short_exposure=3.5,
        idio_vol=0.5,
        margin_response="flag",
        outside_st_gains_annual=50_000.0,
    )
    hit = None
    for seed in range(25):
        calls.clear()
        r = run_path(cfg, seed=seed)
        if r.insolvent:
            hit = r
            break
    assert hit is not None, "no insolvent path found; test setup too tame"
    assert hit.termination_step < cfg.n_steps - 1
    assert max(step for step, _ in calls) == hit.termination_step


def test_run_sweep_parallel_matches_serial():
    cfg = small_cfg()
    serial = run_sweep(cfg, ["100/0", "130/30"], n_paths=6, base_seed=3)
    parallel = run_sweep(cfg, ["100/0", "130/30"], n_paths=6, base_seed=3, n_jobs=2)
    for a, b in zip(serial, parallel, strict=True):
        assert a.book == b.book
        assert a.paths == b.paths
    with pytest.raises(ValueError):
        run_sweep(cfg, ["100/0"], n_paths=1, n_jobs=0)


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
