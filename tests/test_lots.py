import pytest

from talsim.lots import Ledger


def make_ledger(**kwargs):
    return Ledger(steps_per_year=4, **kwargs)


def make_monthly_ledger(**kwargs):
    return Ledger(steps_per_year=12, **kwargs)


def test_open_and_market_value():
    led = make_ledger()
    led.open("long", 0, 10, 100.0, step=0)
    led.open("long", 0, 5, 110.0, step=1)
    assert led.longs.shares_of(0) == 15
    assert led.longs.market_value({0: 120.0}) == 15 * 120.0


def test_hifo_sells_loss_lots_first():
    led = make_ledger()
    led.open("long", 0, 10, 100.0, step=0)  # at 90: loss lot
    led.open("long", 0, 10, 80.0, step=0)  # at 90: gain lot
    realized = led.close("long", 0, 10, 90.0, step=6)
    assert len(realized) == 1
    assert realized[0].basis == pytest.approx(1000.0)
    # The still-held gain lot was bought at step 0, outside the 1-step wash
    # window relative to step 6, so the loss survives intact.
    assert realized[0].gain == pytest.approx(-100.0)


def test_holding_period_character():
    led = make_ledger()
    led.open("long", 0, 10, 100.0, step=0)
    st = led.close("long", 0, 5, 120.0, step=3)
    lt = led.close("long", 0, 5, 120.0, step=4)
    assert st[0].term == "st"
    assert lt[0].term == "lt"


def test_short_gains_are_short_term_and_signed_correctly():
    led = make_ledger()
    led.open("short", 0, 10, 100.0, step=0)
    win = led.close("short", 0, 5, 90.0, step=6)
    lose = led.close("short", 0, 5, 130.0, step=6)
    assert win[0].term == "st" and win[0].gain == pytest.approx(50.0)
    assert lose[0].term == "st" and lose[0].gain == pytest.approx(-150.0)


def test_close_more_than_held_raises():
    led = make_ledger()
    led.open("long", 0, 10, 100.0, step=0)
    with pytest.raises(ValueError):
        led.close("long", 0, 11, 100.0, step=1)


# ---------------------------------------------------------------------------
# Wash-sale enforcement (review blocker 1 regression tests)
# ---------------------------------------------------------------------------


def test_same_day_repurchase_disallows_the_entire_loss():
    """The review's minimal counterexample: sell 10 @ $90 with $100 basis,
    rebuy 10 @ $90 the same step. The loss must be disallowed, added to the
    replacement basis, and the holding period must tack."""
    led = make_ledger()
    led.open("long", 0, 10, 100.0, step=0)
    recs = led.close("long", 0, 10, 90.0, step=5)
    assert recs[0].gain == pytest.approx(-100.0)  # loss real until replaced
    led.open("long", 0, 10, 90.0, step=5)
    assert recs[0].gain == pytest.approx(0.0)  # fully disallowed
    assert recs[0].disallowed == pytest.approx(100.0)
    lot = led.longs.lots[0][0]
    assert lot.basis_per_share == pytest.approx(100.0)  # 90 + 10 transferred
    assert lot.open_step == 0  # tacked
    assert led.disallowed_losses(0, 5) == pytest.approx(100.0)


def test_backward_window_purchase_is_a_replacement():
    """Shares bought within the window BEFORE a loss sale, and still held,
    absorb the wash."""
    led = make_ledger()
    led.open("long", 0, 10, 100.0, step=0)
    led.open("long", 0, 10, 95.0, step=5)  # within window of the sale below
    recs = led.close("long", 0, 10, 90.0, step=5)  # HIFO closes the 100 lot
    assert recs[0].basis == pytest.approx(1000.0)
    assert recs[0].gain == pytest.approx(0.0)
    assert recs[0].disallowed == pytest.approx(100.0)
    survivor = led.longs.lots[0][0]
    assert survivor.basis_per_share == pytest.approx(95.0 + 10.0)
    assert survivor.open_step == 0


def test_repurchase_outside_window_keeps_the_loss():
    led = make_ledger()
    led.open("long", 0, 10, 100.0, step=0)
    recs = led.close("long", 0, 10, 90.0, step=4)
    led.open("long", 0, 10, 90.0, step=6)  # window is 1 step at quarterly
    assert recs[0].gain == pytest.approx(-100.0)
    assert recs[0].disallowed == 0.0


def test_partial_replacement_disallows_proportionally():
    led = make_ledger()
    led.open("long", 0, 10, 100.0, step=0)
    recs = led.close("long", 0, 10, 90.0, step=5)
    led.open("long", 0, 4, 90.0, step=5)  # only 4 replacement shares
    assert recs[0].disallowed == pytest.approx(40.0)
    assert recs[0].gain == pytest.approx(-60.0)


def test_short_side_wash_applies_to_reshorts():
    led = make_ledger()
    led.open("short", 0, 10, 100.0, step=0)
    recs = led.close("short", 0, 10, 110.0, step=5)  # cover higher = loss 100
    led.open("short", 0, 10, 110.0, step=5)  # immediate re-short
    assert recs[0].gain == pytest.approx(0.0)
    assert recs[0].disallowed == pytest.approx(100.0)
    # The deferred loss LOWERS the replacement short's basis (its sale
    # proceeds), so covering later at 110 recognizes the deferred $100
    # loss rather than a phantom zero.
    lot = led.shorts.lots[0][0]
    assert lot.basis_per_share == pytest.approx(100.0)
    later = led.close("short", 0, 10, 110.0, step=8)
    assert later[0].gain == pytest.approx(-100.0)


def test_wash_window_rounds_up_for_finer_cadence():
    monthly = Ledger(steps_per_year=12, wash_window_days=30)
    assert monthly.wash_window_steps == 1
    weekly = Ledger(steps_per_year=52, wash_window_days=30)
    assert weekly.wash_window_steps == 5  # ceil(30 / 7.02)


def test_pil_capitalized_only_within_45_days():
    """Pub 550: payments in lieu raise cover basis only when the short is
    closed by day 45; longer-held shorts get no tax benefit here."""
    led = make_ledger()
    led.open("short", 0, 10, 100.0, step=0)
    led.shorts.lots[0][0].pil_accrued = 20.0
    # Same-step close: held 0 days, capitalize: gain 50 - 20 = 30.
    recs = led.close("short", 0, 10, 95.0, step=0)
    assert recs[0].gain == pytest.approx(30.0)

    led2 = make_ledger()
    led2.open("short", 0, 10, 100.0, step=0)
    led2.shorts.lots[0][0].pil_accrued = 20.0
    # One quarterly step is ~91 days > 45: no capitalization.
    recs = led2.close("short", 0, 10, 95.0, step=1)
    assert recs[0].gain == pytest.approx(50.0)


def test_partial_wash_splits_replacement_lot():
    """The reviewer's counterexample. Buy 5 @ $100; later buy 10 @ $90;
    sell the original 5 @ $90. Only five replacement shares absorb the
    $50 wash: they carry basis $100 and the old holding period, while the
    other five keep basis $90 and the new date. A later 5-share sale at
    $95 must recognize a $25 long-term loss, not zero."""
    led = make_ledger()
    led.open("long", 0, 5, 100.0, step=0)
    led.open("long", 0, 10, 90.0, step=4)
    recs = led.close("long", 0, 5, 90.0, step=4)  # HIFO sells the $100 lot
    assert recs[0].basis == pytest.approx(500.0)
    assert recs[0].disallowed == pytest.approx(50.0)
    assert recs[0].gain == pytest.approx(0.0)

    lots = sorted(led.longs.lots[0], key=lambda lot: lot.basis_per_share)
    assert len(lots) == 2
    assert lots[0].shares == pytest.approx(5) and lots[0].basis_per_share == pytest.approx(90.0)
    assert lots[0].open_step == 4 and not lots[0].was_replacement
    assert lots[1].shares == pytest.approx(5) and lots[1].basis_per_share == pytest.approx(100.0)
    assert lots[1].open_step == 0 and lots[1].was_replacement

    later = led.close("long", 0, 5, 95.0, step=8)  # HIFO picks the loss lot
    assert later[0].gain == pytest.approx(-25.0)
    assert later[0].term == "lt"  # tacked holding period: 8 steps from 0


def test_oversized_replacement_lot_partial_match():
    """A 3-share loss against a 10-share replacement matches only 3."""
    led = make_ledger()
    led.open("long", 0, 3, 100.0, step=0)
    recs = led.close("long", 0, 3, 90.0, step=5)
    led.open("long", 0, 10, 90.0, step=5)
    assert recs[0].disallowed == pytest.approx(30.0)
    lots = sorted(led.longs.lots[0], key=lambda lot: lot.basis_per_share)
    assert lots[0].shares == pytest.approx(7)
    assert lots[0].basis_per_share == pytest.approx(90.0)
    assert lots[1].shares == pytest.approx(3)
    assert lots[1].basis_per_share == pytest.approx(100.0)


def test_wash_matching_is_chronological_across_purchases():
    """Two purchases inside the window: the earlier one absorbs first."""
    led = make_ledger()
    led.open("long", 0, 10, 100.0, step=0)
    recs = led.close("long", 0, 10, 90.0, step=5)
    led.open("long", 0, 4, 91.0, step=5)
    led.open("long", 0, 4, 92.0, step=6)
    assert recs[0].washed_shares == pytest.approx(8)
    by_basis = sorted(led.longs.lots[0], key=lambda lot: lot.basis_per_share)
    # 91-basis lot bought first: fully matched (+10 transfer).
    assert by_basis[0].basis_per_share == pytest.approx(101.0)
    assert by_basis[1].basis_per_share == pytest.approx(102.0)


def test_monthly_cadence_uses_days_not_steps():
    monthly = make_monthly_ledger()
    monthly.open("long", 0, 10, 100.0, step=0)
    st = monthly.close("long", 0, 5, 120.0, step=11)  # ~334 days
    lt = monthly.close("long", 0, 5, 120.0, step=12)  # ~365 days
    assert st[0].term == "st"
    assert lt[0].term == "lt"


def test_policy_queries():
    led = make_ledger()
    led.open("long", 0, 10, 100.0, step=4)
    assert led.recent_open_shares("long", 0, step=5) == pytest.approx(10)
    assert led.recent_open_shares("long", 0, step=7) == 0.0
    led.close("long", 0, 10, 90.0, step=8)
    assert led.loss_sale_blocked("long", 0, step=9)
    assert not led.loss_sale_blocked("long", 0, step=11)
    assert not led.loss_sale_blocked("short", 0, step=9)
