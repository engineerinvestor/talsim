"""Property-based invariants for the ledger.

The wash-sale rule defers value; it never creates or destroys it. Whatever
sequence of trades the policy produces, total economic profit must equal
realized gain plus unrealized gain, and shares must be conserved.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from talsim.lots import Ledger

PRICES = st.floats(min_value=5.0, max_value=200.0)


@st.composite
def trade_sequences(draw):
    """Random open/close sequences on one asset, one side, valid by construction."""
    side = draw(st.sampled_from(["long", "short"]))
    n_events = draw(st.integers(min_value=1, max_value=25))
    events = []
    held = 0.0
    for _ in range(n_events):
        price = draw(PRICES)
        step = len(events)  # strictly increasing steps
        if held > 0.5 and draw(st.booleans()):
            fraction = draw(st.floats(min_value=0.1, max_value=1.0))
            shares = held * fraction
            events.append(("close", shares, price, step))
            held -= shares
        else:
            shares = draw(st.floats(min_value=0.5, max_value=50.0))
            events.append(("open", shares, price, step))
            held += shares
    return side, events


@given(trade_sequences())
@settings(max_examples=200, deadline=None)
def test_wash_adjustments_conserve_total_pnl(seq):
    """Realized gain (post-wash) + unrealized gain == cash P&L of the
    round trips, marking open inventory to the last price."""
    side, events = seq
    led = Ledger(steps_per_year=4)
    cash = 0.0
    sign = 1.0 if side == "long" else -1.0
    last_price = 100.0
    for kind, shares, price, step in events:
        last_price = price
        if kind == "open":
            led.open(side, 0, shares, price, step)
            cash -= sign * shares * price
        else:
            led.close(side, 0, shares, price, step)
            cash += sign * shares * price

    realized = sum(rec.gain for rec in led.realized)
    unrealized = led.book(side).unrealized_gain(0, last_price)
    open_value = sign * led.book(side).shares_of(0) * last_price
    total_pnl = cash + open_value
    assert abs((realized + unrealized) - total_pnl) < 1e-6 * max(1.0, abs(total_pnl))


@given(trade_sequences())
@settings(max_examples=200, deadline=None)
def test_shares_conserved_through_splits(seq):
    """Wash-sale lot splitting must never create or destroy shares."""
    side, events = seq
    led = Ledger(steps_per_year=4)
    opened = closed = 0.0
    for kind, shares, price, step in events:
        if kind == "open":
            led.open(side, 0, shares, price, step)
            opened += shares
        else:
            led.close(side, 0, shares, price, step)
            closed += shares
    held = led.book(side).shares_of(0)
    assert abs(held - (opened - closed)) < 1e-6


@given(trade_sequences())
@settings(max_examples=200, deadline=None)
def test_disallowed_losses_never_negative_and_bounded(seq):
    side, events = seq
    led = Ledger(steps_per_year=4)
    for kind, shares, price, step in events:
        if kind == "open":
            led.open(side, 0, shares, price, step)
        else:
            led.close(side, 0, shares, price, step)
    for rec in led.realized:
        assert rec.disallowed >= 0
        assert rec.washed_shares <= rec.shares + 1e-9
        # Post-adjustment gain of a loss sale can rise at most to zero.
        if rec.disallowed > 0:
            assert rec.gain <= 1e-6
