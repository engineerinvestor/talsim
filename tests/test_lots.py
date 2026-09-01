import pytest

from talsim.lots import LotBook, WashBlocker


def test_open_and_market_value():
    book = LotBook("long", steps_per_year=4)
    book.open(0, 10, 100.0, step=0)
    book.open(0, 5, 110.0, step=1)
    assert book.shares_of(0) == 15
    assert book.market_value({0: 120.0}) == 15 * 120.0


def test_hifo_sells_loss_lots_first():
    book = LotBook("long", steps_per_year=4)
    book.open(0, 10, 100.0, step=0)  # at 90: loss lot
    book.open(0, 10, 80.0, step=0)  # at 90: gain lot
    realized = book.close(0, 10, 90.0, step=1)
    assert len(realized) == 1
    assert realized[0].basis == pytest.approx(1000.0)
    assert realized[0].gain == pytest.approx(-100.0)


def test_partial_close_spans_lots_and_defers_gains():
    book = LotBook("long", steps_per_year=4)
    book.open(0, 10, 100.0, step=0)
    book.open(0, 10, 80.0, step=0)
    realized = book.close(0, 15, 90.0, step=1)
    # All 10 loss shares first, then only 5 of the gain lot.
    assert sum(r.shares for r in realized) == pytest.approx(15)
    assert realized[0].gain == pytest.approx(-100.0)
    assert realized[1].gain == pytest.approx(5 * 10.0)
    assert book.shares_of(0) == pytest.approx(5)


def test_holding_period_character():
    book = LotBook("long", steps_per_year=4)
    book.open(0, 10, 100.0, step=0)
    st = book.close(0, 5, 120.0, step=3)  # 3 steps < 4 => short-term
    lt = book.close(0, 5, 120.0, step=4)  # 4 steps >= 4 => long-term
    assert st[0].term == "st"
    assert lt[0].term == "lt"


def test_short_gains_are_short_term_and_signed_correctly():
    book = LotBook("short", steps_per_year=4)
    book.open(0, 10, 100.0, step=0)  # sold short at 100
    win = book.close(0, 5, 90.0, step=6)  # cover lower = gain
    lose = book.close(0, 5, 130.0, step=6)
    assert win[0].term == "st" and win[0].gain == pytest.approx(50.0)
    assert lose[0].term == "st" and lose[0].gain == pytest.approx(-150.0)


def test_close_more_than_held_raises():
    book = LotBook("long", steps_per_year=4)
    book.open(0, 10, 100.0, step=0)
    with pytest.raises(ValueError):
        book.close(0, 11, 100.0, step=1)


def test_wash_blocker_blocks_then_expires():
    wash = WashBlocker(block_steps=1)
    wash.record_loss_sale("long", 3, step=5)
    assert wash.is_blocked("long", 3, step=5)
    assert not wash.is_blocked("long", 3, step=6)
    assert not wash.is_blocked("short", 3, step=5)  # sides are separate groups
