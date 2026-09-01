import pytest

from talsim.lots import Ledger


def make_ledger(**kwargs):
    return Ledger(steps_per_year=4, **kwargs)


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


def test_wash_window_rounds_up_for_finer_cadence():
    monthly = Ledger(steps_per_year=12, wash_window_days=30)
    assert monthly.wash_window_steps == 1
    weekly = Ledger(steps_per_year=52, wash_window_days=30)
    assert weekly.wash_window_steps == 5  # ceil(30 / 7.02)


def test_short_close_extra_basis_reduces_gain():
    """Accrued payments in lieu raise the basis of shares used to close."""
    led = make_ledger()
    led.open("short", 0, 10, 100.0, step=0)
    recs = led.close("short", 0, 10, 95.0, step=6, extra_basis_per_share=2.0)
    # Gain would be 50 without the PIL basis adjustment; 10 * 2 reduces it.
    assert recs[0].gain == pytest.approx(30.0)


def test_policy_queries():
    led = make_ledger()
    led.open("long", 0, 10, 100.0, step=4)
    assert led.recent_open_shares("long", 0, step=5) == pytest.approx(10)
    assert led.recent_open_shares("long", 0, step=7) == 0.0
    led.close("long", 0, 10, 90.0, step=8)
    assert led.loss_sale_blocked("long", 0, step=9)
    assert not led.loss_sale_blocked("long", 0, step=11)
    assert not led.loss_sale_blocked("short", 0, step=9)
