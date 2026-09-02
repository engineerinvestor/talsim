"""Tax-lot ledger with HIFO selection and enforced wash-sale accounting.

The ledger is independent of the trading policy and is the source of truth
for realized gains: it accepts any sequence of trades, including
noncompliant ones, and applies wash-sale disallowance itself. The policy
layer separately tries to avoid wash sales; the ledger guarantees that a
violation never manufactures a deductible loss.

Wash-sale mechanics, per the shape of IRC 1091 and IRS Publication 550:

- A loss is disallowed to the extent substantially identical shares are
  acquired within the window before or after the sale (the statute's 30
  days; here a configurable number of steps derived from the simulation
  cadence, always rounded up so the model never under-blocks).
- Matching is share-for-share in acquisition order. When only part of a
  replacement lot matches, the lot is SPLIT: the matched shares become
  their own lot carrying the transferred basis and the tacked holding
  period, while the unmatched shares keep their original basis and date.
- A share can serve as a replacement only once.

Payments in lieu of dividends on short positions accrue per lot. When a
short is closed, its accrued payments in lieu are capitalized into the
basis of the shares used to close only if the short was held 45 days or
less (Pub 550); payments on longer-held shorts get no tax benefit here,
a deliberate conservatism until an investment-interest deduction bucket
(with its own limitations) exists.

Simplifications, stated plainly:

- Each (side, asset) pair is its own wash group. Cross-asset
  "substantially identical" determinations and household scope (spouse,
  IRA, controlled entities) are out of scope.
- Short-sale gains and losses are treated as short-term (the common case
  under Pub 550's delivered-shares rule).
- A replacement bought after year-end settlement still adjusts the loss
  record; totals recomputed from records remain exact, while the
  already-settled year is not reopened. The compliant policy layer makes
  this case unreachable in shipped experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

PIL_CAPITALIZATION_MAX_DAYS = 45.0


@dataclass
class Lot:
    asset: int
    shares: float  # always positive; side is carried by the book
    basis_per_share: float  # for shorts, the sale price received at open
    open_step: int
    # True once this lot's shares have served as wash-sale replacements;
    # a share can absorb a wash only once.
    was_replacement: bool = False
    # Accrued payments in lieu of dividends (short lots only).
    pil_accrued: float = 0.0


@dataclass
class Realized:
    asset: int
    side: str  # "long" | "short"
    shares: float
    proceeds: float
    basis: float
    gain: float  # after any wash disallowance
    term: str  # "st" | "lt"
    step: int
    open_step: int
    disallowed: float = 0.0  # wash-disallowed loss, stored positive
    washed_shares: float = 0.0  # loss shares already matched to replacements

    def unwashed_loss_shares(self) -> float:
        pre_gain = self.gain - self.disallowed
        if pre_gain >= 0:
            return 0.0
        return self.shares - self.washed_shares

    def loss_per_share(self) -> float:
        pre_gain = self.gain - self.disallowed
        return -pre_gain / self.shares if pre_gain < 0 else 0.0


@dataclass
class LotBook:
    """Inventory for one side (long or short) across all assets.

    Lot lists remain in acquisition order at all times; HIFO share
    selection sorts a view, never the list itself, because wash-sale
    matching must walk purchases chronologically.
    """

    side: str  # "long" | "short"
    steps_per_year: int
    lots: dict[int, list[Lot]] = field(default_factory=dict)

    def shares_of(self, asset: int) -> float:
        return sum(lot.shares for lot in self.lots.get(asset, []))

    def market_value(self, prices) -> float:
        return sum(lot.shares * prices[asset] for asset, lots in self.lots.items() for lot in lots)

    def unrealized_gain(self, asset: int, price: float) -> float:
        total = 0.0
        for lot in self.lots.get(asset, []):
            if self.side == "long":
                total += lot.shares * (price - lot.basis_per_share)
            else:
                total += lot.shares * (lot.basis_per_share - price)
        return total


class Ledger:
    """Both books plus the realized-trade record and wash enforcement."""

    def __init__(self, steps_per_year: int, wash_window_days: int = 30) -> None:
        self.steps_per_year = steps_per_year
        self.step_days = 365.0 / steps_per_year
        # Round up: the model may over-block, never under-block.
        self.wash_window_steps = max(1, math.ceil(wash_window_days / self.step_days))
        self.longs = LotBook("long", steps_per_year)
        self.shorts = LotBook("short", steps_per_year)
        self.realized: list[Realized] = []

    def book(self, side: str) -> LotBook:
        return self.longs if side == "long" else self.shorts

    def held_days(self, lot: Lot, step: int) -> float:
        return (step - lot.open_step) * self.step_days

    # ------------------------------------------------------------------
    # Trade entry points
    # ------------------------------------------------------------------

    def open(self, side: str, asset: int, shares: float, price: float, step: int) -> None:
        if shares <= 0:
            raise ValueError("open() requires positive shares")
        lot = Lot(asset, shares, price, step)
        self.book(side).lots.setdefault(asset, []).append(lot)
        # Forward wash: the new purchase replaces recent unmatched loss
        # sales, oldest first. Records are step-ordered, so scan back only
        # to the window edge, then process forward chronologically.
        recent: list[Realized] = []
        for rec in reversed(self.realized):
            if step - rec.step > self.wash_window_steps:
                break
            if rec.asset == asset and rec.side == side:
                recent.append(rec)
        for rec in reversed(recent):
            if lot.shares <= 1e-12 or lot.was_replacement:
                break
            if rec.unwashed_loss_shares() <= 1e-12:
                continue
            lot = self._match(side, asset, lot, rec)

    def close(
        self,
        side: str,
        asset: int,
        shares: float,
        price: float,
        step: int,
        prefer_gains: bool = False,
    ) -> list[Realized]:
        """Close up to `shares` using tax-minimizing (HIFO-style) selection.

        `prefer_gains` reverses the selection order: a risk-driven
        reduction of a recently bought position sells gain lots first, so
        it realizes washable losses only when it runs out of gains.
        """
        if shares <= 0:
            raise ValueError("close() requires positive shares")
        book = self.book(side)
        inventory = book.lots.get(asset, [])
        if not inventory:
            raise KeyError(f"no {side} lots for asset {asset}")

        def pil_ps(lot: Lot) -> float:
            """Capitalizable payments in lieu per share (shorts, <=45 days)."""
            if side != "short" or lot.shares <= 0 or lot.pil_accrued <= 0:
                return 0.0
            if self.held_days(lot, step) > PIL_CAPITALIZATION_MAX_DAYS:
                return 0.0
            return lot.pil_accrued / lot.shares

        def per_share_gain(lot: Lot) -> float:
            if side == "long":
                return price - lot.basis_per_share
            return lot.basis_per_share - price - pil_ps(lot)

        # Selection order is a sorted VIEW; the inventory list itself stays
        # in acquisition order for wash chronology.
        order = sorted(inventory, key=per_share_gain, reverse=prefer_gains)
        out: list[Realized] = []
        remaining = shares
        for lot in order:
            if remaining <= 1e-12:
                break
            take = min(lot.shares, remaining)
            extra_ps = pil_ps(lot)
            gain = take * per_share_gain(lot)
            if side == "long":
                proceeds = take * price
                basis = take * lot.basis_per_share
                term = "lt" if self.held_days(lot, step) >= 365.0 else "st"
            else:
                proceeds = take * lot.basis_per_share
                basis = take * (price + extra_ps)
                term = "st"
            rec = Realized(asset, side, take, proceeds, basis, gain, term, step, lot.open_step)
            # Accrued PIL leaves with the shares whether or not capitalized.
            if lot.shares > 0:
                lot.pil_accrued *= max(0.0, 1 - take / lot.shares)
            lot.shares -= take
            remaining -= take
            self.realized.append(rec)
            out.append(rec)
        book.lots[asset] = [lot for lot in inventory if lot.shares > 1e-12]
        if remaining > 1e-9:
            raise ValueError(
                f"tried to close {shares} shares of asset {asset}, only {shares - remaining} held"
            )
        # Wash matching runs after the whole close: a mid-loop lot split
        # would move shares into lots invisible to the sorted selection
        # view above. All of this close's sales share one step, so the
        # deferred matching is chronologically identical.
        for rec in out:
            if rec.gain < 0:
                self._wash_backward(side, asset, rec, step)
        return out

    # ------------------------------------------------------------------
    # Wash-sale enforcement
    # ------------------------------------------------------------------

    def _wash_backward(self, side: str, asset: int, rec: Realized, step: int) -> None:
        """A fresh loss looks back: shares bought within the window and
        still held are replacements, matched in acquisition order."""
        inventory = self.book(side).lots.get(asset, [])
        # Walk a snapshot in acquisition order; _match may split lots and
        # insert into the inventory, which must not disturb this walk.
        for lot in list(inventory):
            if rec.unwashed_loss_shares() <= 1e-12:
                break
            if lot.shares <= 1e-12 or lot.was_replacement:
                continue
            if step - lot.open_step > self.wash_window_steps:
                continue
            self._match(side, asset, lot, rec)

    def _match(self, side: str, asset: int, lot: Lot, rec: Realized) -> Lot:
        """Match up to a full lot against a loss record.

        On a partial match the lot is split: the matched shares become a
        new lot with transferred basis and tacked holding period; the
        remainder keeps its original basis and date. Returns the lot that
        still holds unmatched shares (for the forward-wash caller).
        """
        loss_ps = rec.loss_per_share()
        match = min(lot.shares, rec.unwashed_loss_shares())
        if match <= 1e-12 or loss_ps <= 0:
            return lot
        rec.gain += loss_ps * match
        rec.disallowed += loss_ps * match
        rec.washed_shares += match

        # The deferred loss moves into the replacement's basis so it is
        # recognized when the replacement closes. For longs that means a
        # HIGHER cost basis; for shorts, "basis" is the sale proceeds, so
        # the deferred loss LOWERS it (a higher short basis would turn the
        # deferred loss into a phantom future gain).
        basis_shift = loss_ps if side == "long" else -loss_ps

        inventory = self.book(side).lots.setdefault(asset, [])
        if match >= lot.shares - 1e-12:
            lot.basis_per_share += basis_shift
            lot.open_step = min(lot.open_step, rec.open_step)
            lot.was_replacement = True
            return lot
        # Split: the matched sublot inherits a proportional share of any
        # accrued payments in lieu.
        matched = Lot(
            asset,
            match,
            lot.basis_per_share + basis_shift,
            min(lot.open_step, rec.open_step),
            was_replacement=True,
            pil_accrued=lot.pil_accrued * match / lot.shares,
        )
        lot.pil_accrued -= matched.pil_accrued
        lot.shares -= match
        inventory.insert(inventory.index(lot), matched)
        return lot

    # ------------------------------------------------------------------
    # Policy-facing queries (avoidance, not enforcement)
    # ------------------------------------------------------------------

    def recent_open_shares(self, side: str, asset: int, step: int) -> float:
        return sum(
            lot.shares
            for lot in self.book(side).lots.get(asset, [])
            if step - lot.open_step <= self.wash_window_steps and not lot.was_replacement
        )

    def loss_sale_blocked(self, side: str, asset: int, step: int) -> bool:
        for rec in reversed(self.realized):
            if step - rec.step > self.wash_window_steps:
                break
            if rec.asset == asset and rec.side == side and rec.gain - rec.disallowed < 0:
                return True
        return False

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def realized_totals(self, from_step: int, to_step: int) -> tuple[float, float]:
        st = lt = 0.0
        for rec in self.realized:
            if from_step <= rec.step <= to_step:
                if rec.term == "st":
                    st += rec.gain
                else:
                    lt += rec.gain
        return st, lt

    def gross_losses(self, from_step: int, to_step: int) -> float:
        return sum(
            -rec.gain for rec in self.realized if from_step <= rec.step <= to_step and rec.gain < 0
        )

    def disallowed_losses(self, from_step: int, to_step: int) -> float:
        return sum(rec.disallowed for rec in self.realized if from_step <= rec.step <= to_step)

    def is_empty(self) -> bool:
        return all(
            not lots or all(lot.shares <= 1e-9 for lot in lots)
            for book in (self.longs, self.shorts)
            for lots in book.lots.values()
        )
