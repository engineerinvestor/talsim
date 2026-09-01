"""Tax-lot inventory with HIFO selection and wash-sale re-entry blocking.

The ledger is deliberately independent of the optimizer: it accepts any
sequence of trades and reports realized gains with short/long-term character,
so a trade list from any policy can be audited by the same accounting.

Simplifications, stated plainly:

- Each asset is its own wash-sale group. Real substantially-identical
  determinations (share classes, near-identical ETFs) are out of scope.
- After a loss sale, re-entry on the same side is blocked for a configurable
  number of steps. At the default quarterly cadence one blocked step is 91
  days, which over-blocks relative to the statute's 30 days; that error is
  conservative (it can only understate harvesting, never invent it).
- Short-sale gains and losses are treated as short-term. Publication 550's
  character rules key off the holding period of the shares delivered to
  close, which is typically moments; the long-term edge cases (holding
  substantially identical shares for over a year) do not arise in this model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Lot:
    asset: int
    shares: float  # always positive; side is carried by the book
    basis_per_share: float  # for shorts, the sale price received at open
    open_step: int


@dataclass
class Realized:
    asset: int
    side: str  # "long" | "short"
    shares: float
    proceeds: float
    basis: float
    gain: float
    term: str  # "st" | "lt"
    step: int


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

    def open(self, asset: int, shares: float, price: float, step: int) -> None:
        if shares <= 0:
            raise ValueError("open() requires positive shares")
        self.lots.setdefault(asset, []).append(Lot(asset, shares, price, step))

    def unrealized_gain(self, asset: int, price: float) -> float:
        total = 0.0
        for lot in self.lots.get(asset, []):
            if self.side == "long":
                total += lot.shares * (price - lot.basis_per_share)
            else:
                total += lot.shares * (lot.basis_per_share - price)
        return total

    def close(self, asset: int, shares: float, price: float, step: int) -> list[Realized]:
        """Close up to `shares` using tax-minimizing (HIFO-style) selection.

        Loss lots go first, biggest per-share loss first; gain lots go last,
        smallest per-share gain first. Within the model this both harvests
        losses and defers gains whenever a partial close allows it.
        """
        if shares <= 0:
            raise ValueError("close() requires positive shares")
        inventory = self.lots.get(asset, [])
        if not inventory:
            raise KeyError(f"no {self.side} lots for asset {asset}")

        def per_share_gain(lot: Lot) -> float:
            if self.side == "long":
                return price - lot.basis_per_share
            return lot.basis_per_share - price

        inventory.sort(key=per_share_gain)
        realized: list[Realized] = []
        remaining = shares
        while remaining > 1e-12 and inventory:
            lot = inventory[0]
            take = min(lot.shares, remaining)
            gain = take * per_share_gain(lot)
            if self.side == "long":
                proceeds = take * price
                basis = take * lot.basis_per_share
                held_steps = step - lot.open_step
                term = "lt" if held_steps >= self.steps_per_year else "st"
            else:
                proceeds = take * lot.basis_per_share
                basis = take * price
                term = "st"  # simplification documented in the module docstring
            realized.append(Realized(asset, self.side, take, proceeds, basis, gain, term, step))
            lot.shares -= take
            remaining -= take
            if lot.shares <= 1e-12:
                inventory.pop(0)
        if remaining > 1e-9:
            raise ValueError(
                f"tried to close {shares} shares of asset {asset}, only {shares - remaining} held"
            )
        return realized


@dataclass
class WashBlocker:
    """Tracks re-entry blocks after loss sales, one wash group per asset."""

    block_steps: int = 1
    blocked_until: dict[tuple[str, int], int] = field(default_factory=dict)

    def record_loss_sale(self, side: str, asset: int, step: int) -> None:
        key = (side, asset)
        until = step + self.block_steps
        self.blocked_until[key] = max(self.blocked_until.get(key, 0), until)

    def is_blocked(self, side: str, asset: int, step: int) -> bool:
        return self.blocked_until.get((side, asset), 0) > step
