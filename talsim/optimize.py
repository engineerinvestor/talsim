"""Target weights and tax-aware trade planning.

The policy layer is separate from the accounting layer: it produces a plain
trade list that the `Ledger` audits and, if necessary, corrects for wash
sales. The policy tries to be compliant on its own; the ledger guarantees
correctness either way.

Trades are constructed from desired post-trade state per side, never from
signed drift, so a short-to-long transition lands exactly on its target
instead of double-counting the cover. Three tax-aware rules filter the
path from current state to target:

1. Harvest: a lot whose price sits more than the threshold below basis is
   realized, unless shares of the same (side, asset) were acquired within
   the wash window, in which case harvesting would be immediately washed
   and is skipped.
2. Defer: reductions that would realize a net gain are skipped while the
   position's drift stays inside twice the rebalance band.
3. Block: after a loss sale, additions to the same (side, asset) wait out
   the wash window, and nothing harvested is repurchased in the same step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ScenarioConfig
from .lots import Ledger, Realized


def target_weights(signal: np.ndarray, long_exposure: float, short_exposure: float) -> np.ndarray:
    """Signal-tilted weights summing to net exposure with exact gross targets.

    Long-only books get no tilt at all: the 100/0 baseline in a leverage
    comparison is a passive equal-weight harvesting portfolio, and its
    tracking error should come from harvest-and-block deviations, not bets.
    """
    k = len(signal)
    base = np.full(k, (long_exposure - short_exposure) / k)
    if short_exposure == 0:
        return np.full(k, long_exposure / k)

    ranks = np.argsort(np.argsort(signal))  # 0..k-1, higher = stronger signal
    tilt = ranks - ranks.mean()
    scale_grid = np.linspace(0.1, 50.0, 500)
    best = base
    best_err = np.inf
    for scale in scale_grid:
        w = base + tilt / np.abs(tilt).sum() * scale
        short_sum = -w[w < 0].sum()
        err = abs(short_sum - short_exposure)
        if err < best_err:
            best_err = err
            best = w
    w = best.copy()
    neg = w < 0
    if w[neg].sum() != 0:
        w[neg] *= short_exposure / (-w[neg].sum())
    pos = ~neg
    w[pos] *= long_exposure / w[pos].sum()
    return w


@dataclass
class TradePlan:
    buys: list[tuple[int, float]] = field(default_factory=list)
    long_sells: list[tuple[int, float]] = field(default_factory=list)
    short_opens: list[tuple[int, float]] = field(default_factory=list)
    short_covers: list[tuple[int, float]] = field(default_factory=list)
    # (side, asset) reductions that should sell gain lots first because a
    # recent purchase would wash any realized loss.
    prefer_gains: set[tuple[str, int]] = field(default_factory=set)


def plan_trades(
    cfg: ScenarioConfig,
    targets: np.ndarray,  # weights, fraction of NAV
    nav: float,
    prices: np.ndarray,
    ledger: Ledger,
    step: int,
) -> TradePlan:
    plan = TradePlan()
    band_dollars = cfg.rebalance_band * nav
    n = cfg.n_assets

    # ------------------------------------------------------------------
    # Pass 1: per-asset harvest decisions and wash caps.
    # ------------------------------------------------------------------
    harvest_shares = {("long", a): 0.0 for a in range(n)} | {("short", a): 0.0 for a in range(n)}
    capped: dict[tuple[str, int], bool] = {}
    desired_dollars: dict[tuple[str, int], float] = {}
    target_dollars: dict[tuple[str, int], float] = {}

    for asset in range(n):
        price = prices[asset]
        w = targets[asset]
        side_targets = {
            "long": max(w, 0.0) * nav,
            "short": max(-w, 0.0) * nav,
        }
        for side in ("long", "short"):
            book = ledger.book(side)
            current = book.shares_of(asset)
            target = side_targets[side]
            target_dollars[(side, asset)] = target
            if current <= 0 and target <= 0:
                capped[(side, asset)] = False
                desired_dollars[(side, asset)] = 0.0
                continue

            harvest = 0.0
            if ledger.recent_open_shares(side, asset, step) <= 1e-9:
                for lot in book.lots.get(asset, []):
                    is_long_loss = side == "long" and price < lot.basis_per_share * (
                        1 - cfg.harvest_threshold
                    )
                    is_short_loss = side == "short" and price > lot.basis_per_share * (
                        1 + cfg.harvest_threshold
                    )
                    if is_long_loss or is_short_loss:
                        harvest += lot.shares
            harvest_shares[(side, asset)] = harvest
            post_harvest = current - harvest

            is_capped = harvest > 0 or ledger.loss_sale_blocked(side, asset, step)
            capped[(side, asset)] = is_capped
            desired = target if not is_capped else min(target, post_harvest * price)
            desired_dollars[(side, asset)] = desired

    # ------------------------------------------------------------------
    # Pass 1.5: keep each side above its harvest exposure floor. When most
    # of a side is at a loss simultaneously, harvesting everything would
    # flatten the book for a full wash window; the smallest losses defer.
    # ------------------------------------------------------------------
    for side in ("long", "short"):
        side_target_total = sum(target_dollars[(side, a)] for a in range(n))
        if side_target_total <= 0:
            continue
        floor = cfg.harvest_exposure_floor * side_target_total

        def achieved_total(s: str = side) -> float:
            return sum(desired_dollars[(s, a)] for a in range(n))

        # Names whose cap comes only from this step's harvest can be
        # un-harvested; names blocked by a prior loss sale cannot.
        revocable = [
            a
            for a in range(n)
            if harvest_shares[(side, a)] > 0 and not ledger.loss_sale_blocked(side, a, step)
        ]

        # Defer the smallest harvest losses first.
        def harvest_loss(a: int, s: str = side) -> float:
            book = ledger.book(s)
            price = prices[a]
            loss = 0.0
            for lot in book.lots.get(a, []):
                if s == "long" and price < lot.basis_per_share * (1 - cfg.harvest_threshold):
                    loss += lot.shares * (lot.basis_per_share - price)
                elif s == "short" and price > lot.basis_per_share * (1 + cfg.harvest_threshold):
                    loss += lot.shares * (price - lot.basis_per_share)
            return loss

        revocable.sort(key=harvest_loss)
        while achieved_total() < floor and revocable:
            a = revocable.pop(0)
            harvest_shares[(side, a)] = 0.0
            capped[(side, a)] = False
            desired_dollars[(side, a)] = target_dollars[(side, a)]

    # ------------------------------------------------------------------
    # Pass 2: redistribute wash-blocked exposure to substitute names, so a
    # blocked short does not silently turn a 200/100 book into 200/30.
    # ------------------------------------------------------------------
    for side in ("long", "short"):
        side_target_total = sum(target_dollars[(side, a)] for a in range(n))
        achieved = sum(desired_dollars[(side, a)] for a in range(n))
        deficit = side_target_total - achieved
        if deficit <= band_dollars:
            continue
        open_names = [
            a for a in range(n) if not capped[(side, a)] and target_dollars[(side, a)] > 0
        ]
        basis_total = sum(target_dollars[(side, a)] for a in open_names)
        if basis_total <= 0:
            continue
        for a in open_names:
            desired_dollars[(side, a)] += deficit * target_dollars[(side, a)] / basis_total

    # ------------------------------------------------------------------
    # Pass 3: turn desired state into trades with band and deferral rules.
    # ------------------------------------------------------------------
    for asset in range(n):
        price = prices[asset]
        for side in ("long", "short"):
            book = ledger.book(side)
            current = book.shares_of(asset)
            harvest = harvest_shares[(side, asset)]
            desired = desired_dollars[(side, asset)] / price
            if current <= 0 and desired <= 0:
                continue
            post_harvest = current - harvest

            delta = desired - post_harvest
            reduce_shares = 0.0
            add_shares = 0.0
            if delta * price > band_dollars:
                add_shares = delta
            elif -delta * price > band_dollars:
                # Large drifts always trade: risk control outranks tax
                # niceties, and the ledger prices any wash correctly. Small
                # drifts defer when the sale would realize a gain or would
                # wash against a recent purchase.
                execute = -delta * price > 2 * band_dollars
                recent = ledger.recent_open_shares(side, asset, step) > 1e-9
                if not execute:
                    gain = book.unrealized_gain(asset, price)
                    if gain <= 0 and not recent:
                        execute = True
                if execute:
                    reduce_shares = min(-delta, post_harvest)
                    if recent:
                        plan.prefer_gains.add((side, asset))

            close_shares = harvest + reduce_shares
            if side == "long":
                if close_shares > 1e-9:
                    plan.long_sells.append((asset, close_shares))
                if add_shares > 1e-9:
                    plan.buys.append((asset, add_shares))
            else:
                if close_shares > 1e-9:
                    plan.short_covers.append((asset, close_shares))
                if add_shares > 1e-9:
                    plan.short_opens.append((asset, add_shares))
    return plan


def execute_plan(
    plan: TradePlan,
    prices: np.ndarray,
    ledger: Ledger,
    step: int,
) -> tuple[list[Realized], float, float]:
    """Execute against the ledger. Returns (realized, traded $, cash delta).

    Closes run before opens so freed cash funds the additions and so the
    ledger sees loss sales before any same-step replacement purchase.
    Payments in lieu accrue on short lots inside the ledger, which applies
    the 45-day capitalization rule itself at close.
    """
    realized: list[Realized] = []
    traded = 0.0
    cash_delta = 0.0
    for asset, shares in plan.long_sells:
        realized.extend(
            ledger.close(
                "long",
                asset,
                shares,
                prices[asset],
                step,
                prefer_gains=("long", asset) in plan.prefer_gains,
            )
        )
        traded += shares * prices[asset]
        cash_delta += shares * prices[asset]
    for asset, shares in plan.short_covers:
        realized.extend(
            ledger.close(
                "short",
                asset,
                shares,
                prices[asset],
                step,
                prefer_gains=("short", asset) in plan.prefer_gains,
            )
        )
        traded += shares * prices[asset]
        cash_delta -= shares * prices[asset]
    for asset, shares in plan.buys:
        ledger.open("long", asset, shares, prices[asset], step)
        traded += shares * prices[asset]
        cash_delta -= shares * prices[asset]
    for asset, shares in plan.short_opens:
        ledger.open("short", asset, shares, prices[asset], step)
        traded += shares * prices[asset]
        cash_delta += shares * prices[asset]
    return realized, traded, cash_delta
