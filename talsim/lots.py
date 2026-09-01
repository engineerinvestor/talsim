"""Tax-lot ledger with HIFO selection and enforced wash-sale accounting.

The ledger is independent of the trading policy and is the source of truth
for realized gains: it accepts any sequence of trades, including
noncompliant ones, and applies wash-sale disallowance itself. The policy
layer separately tries to avoid wash sales; the ledger guarantees that a
violation never manufactures a deductible loss.

Wash-sale mechanics implemented here, per the shape of IRC 1091 and IRS
Publication 550:

- A loss on a sale is disallowed to the extent substantially identical
  shares are acquired within the window before or after the sale (the
  statute's 30 days; here a configurable number of steps derived from the
  simulation cadence, always rounded up so the model never under-blocks).
- The disallowed loss is added to the basis of the replacement shares, and
  the replacement lot's holding period tacks back to the original lot's.
- Matching is share-for-share: a replacement share can absorb wash from at
  most one loss share and vice versa.

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


@dataclass
class Lot:
    asset: int
    shares: float  # always positive; side is carried by the book
    basis_per_share: float  # for shorts, the sale price received at open
    open_step: int
    # Shares of this lot that have already served as wash-sale replacements.
    wash_absorbed: float = 0.0


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


@dataclass
class LotBook:
    """Inventory for one side (long or short) across all assets."""

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
        step_days = 365.0 / steps_per_year
        # Round up: the model may over-block, never under-block.
        self.wash_window_steps = max(1, math.ceil(wash_window_days / step_days))
        self.longs = LotBook("long", steps_per_year)
        self.shorts = LotBook("short", steps_per_year)
        self.realized: list[Realized] = []

    def book(self, side: str) -> LotBook:
        return self.longs if side == "long" else self.shorts

    # ------------------------------------------------------------------
    # Trade entry points
    # ------------------------------------------------------------------

    def open(self, side: str, asset: int, shares: float, price: float, step: int) -> None:
        if shares <= 0:
            raise ValueError("open() requires positive shares")
        lot = Lot(asset, shares, price, step)
        self.book(side).lots.setdefault(asset, []).append(lot)
        self._wash_forward(side, asset, lot, step)

    def close(
        self,
        side: str,
        asset: int,
        shares: float,
        price: float,
        step: int,
        extra_basis_per_share: float = 0.0,
        prefer_gains: bool = False,
    ) -> list[Realized]:
        """Close up to `shares` using tax-minimizing (HIFO-style) selection.

        `extra_basis_per_share` implements the Pub 550 treatment of accrued
        payments in lieu on shorts: it raises the basis of the shares used
        to close, reducing the taxable gain (or deepening the loss).
        `prefer_gains` reverses the selection order: a risk-driven reduction
        of a recently bought position sells gain lots first, so it realizes
        washable losses only when it runs out of gains to sell.
        """
        if shares <= 0:
            raise ValueError("close() requires positive shares")
        book = self.book(side)
        inventory = book.lots.get(asset, [])
        if not inventory:
            raise KeyError(f"no {side} lots for asset {asset}")

        def per_share_gain(lot: Lot) -> float:
            if side == "long":
                return price - lot.basis_per_share
            return lot.basis_per_share - price - extra_basis_per_share

        inventory.sort(key=per_share_gain, reverse=prefer_gains)
        out: list[Realized] = []
        remaining = shares
        while remaining > 1e-12 and inventory:
            lot = inventory[0]
            take = min(lot.shares, remaining)
            gain = take * per_share_gain(lot)
            if side == "long":
                proceeds = take * price
                basis = take * lot.basis_per_share
                held_steps = step - lot.open_step
                term = "lt" if held_steps >= self.steps_per_year else "st"
            else:
                proceeds = take * lot.basis_per_share
                basis = take * (price + extra_basis_per_share)
                term = "st"
            rec = Realized(asset, side, take, proceeds, basis, gain, term, step, lot.open_step)
            lot.shares -= take
            lot.wash_absorbed = min(lot.wash_absorbed, lot.shares)
            remaining -= take
            if lot.shares <= 1e-12:
                inventory.pop(0)
            if rec.gain < 0:
                self._wash_backward(side, asset, rec, step)
            self.realized.append(rec)
            out.append(rec)
        if remaining > 1e-9:
            raise ValueError(
                f"tried to close {shares} shares of asset {asset}, only {shares - remaining} held"
            )
        return out

    # ------------------------------------------------------------------
    # Wash-sale enforcement
    # ------------------------------------------------------------------

    def _wash_backward(self, side: str, asset: int, rec: Realized, step: int) -> None:
        """A fresh loss looks back: shares bought in the window and still
        held are replacements; the matched loss is disallowed and moves
        into the replacement lots' basis."""
        loss_per_share = -rec.gain / rec.shares
        for lot in self.book(side).lots.get(asset, []):
            if rec.washed_shares >= rec.shares - 1e-12:
                break
            if step - lot.open_step > self.wash_window_steps:
                continue
            capacity = lot.shares - lot.wash_absorbed
            if capacity <= 1e-12:
                continue
            match = min(capacity, rec.shares - rec.washed_shares)
            disallow = loss_per_share * match
            rec.gain += disallow
            rec.disallowed += disallow
            rec.washed_shares += match
            lot.basis_per_share += disallow / lot.shares
            lot.wash_absorbed += match
            lot.open_step = min(lot.open_step, rec.open_step)  # holding period tacks

    def _wash_forward(self, side: str, asset: int, lot: Lot, step: int) -> None:
        """A fresh purchase looks back at recent loss sales: it is a
        replacement for any unmatched loss shares sold within the window."""
        for rec in reversed(self.realized):
            if lot.shares - lot.wash_absorbed <= 1e-12:
                break
            if step - rec.step > self.wash_window_steps:
                break  # realized list is step-ordered; older records only
            if rec.asset != asset or rec.side != side:
                continue
            pre_gain = rec.gain - rec.disallowed
            if pre_gain >= 0 or rec.washed_shares >= rec.shares - 1e-12:
                continue
            loss_per_share = -pre_gain / rec.shares
            match = min(lot.shares - lot.wash_absorbed, rec.shares - rec.washed_shares)
            disallow = loss_per_share * match
            rec.gain += disallow
            rec.disallowed += disallow
            rec.washed_shares += match
            lot.basis_per_share += disallow / lot.shares
            lot.wash_absorbed += match
            lot.open_step = min(lot.open_step, rec.open_step)

    # ------------------------------------------------------------------
    # Policy-facing queries (avoidance, not enforcement)
    # ------------------------------------------------------------------

    def recent_open_shares(self, side: str, asset: int, step: int) -> float:
        """Shares of (side, asset) opened within the window and still held."""
        return sum(
            lot.shares
            for lot in self.book(side).lots.get(asset, [])
            if step - lot.open_step <= self.wash_window_steps
        )

    def loss_sale_blocked(self, side: str, asset: int, step: int) -> bool:
        """True when a loss sale of (side, asset) occurred within the window,
        so re-entry would trigger disallowance."""
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
        """(short-term, long-term) realized gain in steps [from_step, to_step],
        net of wash disallowance as currently recorded."""
        st = lt = 0.0
        for rec in self.realized:
            if from_step <= rec.step <= to_step:
                if rec.term == "st":
                    st += rec.gain
                else:
                    lt += rec.gain
        return st, lt

    def gross_losses(self, from_step: int, to_step: int) -> float:
        """Deductible realized losses (post-disallowance) in the step range."""
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
