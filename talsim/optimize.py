"""Target weights and tax-aware trade planning.

The policy layer is intentionally separate from the accounting layer: it
produces a plain trade list (asset, side, shares) that `LotBook` can audit.

The construction is a transparent heuristic rather than a numerical
optimizer. Benchmark weights are equal-weight; the active tilt is
proportional to the demeaned signal rank, scaled to hit the book's exact
long and short exposure targets. Trading toward those targets is filtered
through three tax-aware rules:

1. Harvest: any lot whose price sits more than the threshold below basis is
   realized, regardless of the target, and its wash group is blocked.
2. Defer: sells that would realize net gains are skipped while the position's
   drift from target stays inside the rebalance band.
3. Respect blocks: buys (or short re-entries) in a blocked wash group wait.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ScenarioConfig
from .lots import LotBook, Realized, WashBlocker


def target_weights(signal: np.ndarray, long_exposure: float, short_exposure: float) -> np.ndarray:
    """Signal-tilted weights summing to net exposure with exact gross targets.

    Returns weights as fractions of NAV: positive entries sum to
    `long_exposure`, negative entries sum to `-short_exposure`.
    """
    k = len(signal)
    ranks = np.argsort(np.argsort(signal))  # 0..k-1, higher = stronger signal
    tilt = ranks - ranks.mean()
    base = np.full(k, (long_exposure - short_exposure) / k)

    if short_exposure == 0:
        # Long-only: a mild tilt only. The long-only baseline in a leverage
        # comparison is a harvesting index fund, not an active portfolio, so
        # its tracking error should come mostly from harvest-and-block
        # deviations rather than from deliberate bets.
        raw = base + tilt / (np.abs(tilt).sum()) * long_exposure * 0.15
        raw = np.clip(raw, 0.0, None)
        return raw / raw.sum() * long_exposure

    # Long-short: scale the tilt so shorts sum to the short target.
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
    w = best
    # Exact repair: scale negatives to the short target, positives to the long.
    neg = w < 0
    if w[neg].sum() != 0:
        w[neg] *= short_exposure / (-w[neg].sum())
    pos = ~neg
    w[pos] *= long_exposure / w[pos].sum()
    return w


@dataclass
class TradePlan:
    buys: list[tuple[int, float]]  # (asset, shares) long buys
    long_sells: list[tuple[int, float]]
    short_opens: list[tuple[int, float]]
    short_covers: list[tuple[int, float]]


def plan_trades(
    cfg: ScenarioConfig,
    targets: np.ndarray,  # weights, fraction of NAV
    nav: float,
    prices: np.ndarray,
    longs: LotBook,
    shorts: LotBook,
    wash: WashBlocker,
    step: int,
) -> TradePlan:
    plan = TradePlan([], [], [], [])
    band_dollars = cfg.rebalance_band * nav

    for asset in range(cfg.n_assets):
        price = prices[asset]
        target_dollars = targets[asset] * nav
        long_dollars = longs.shares_of(asset) * price
        short_dollars = shorts.shares_of(asset) * price

        # Harvest pass: realize losses beyond the threshold on either side.
        harvest_long = 0.0
        for lot in list(longs.lots.get(asset, [])):
            if price < lot.basis_per_share * (1 - cfg.harvest_threshold):
                harvest_long += lot.shares
        harvest_short = 0.0
        for lot in list(shorts.lots.get(asset, [])):
            if price > lot.basis_per_share * (1 + cfg.harvest_threshold):
                harvest_short += lot.shares

        current = long_dollars - short_dollars
        drift = target_dollars - current

        if target_dollars >= 0:
            # Want a net long position. Cover any short entirely.
            cover = shorts.shares_of(asset)
            if cover > 0:
                plan.short_covers.append((asset, cover))
            desired_change = drift
            if harvest_long > 0:
                plan.long_sells.append((asset, harvest_long))
                desired_change += harvest_long * price
            if desired_change > band_dollars:
                if not wash.is_blocked("long", asset, step):
                    plan.buys.append((asset, desired_change / price))
            elif desired_change < -band_dollars:
                sell_shares = min(-desired_change / price, longs.shares_of(asset) - harvest_long)
                if sell_shares > 1e-9:
                    # Defer: skip if this sale would realize a net gain and
                    # the drift is inside twice the band.
                    gain = longs.unrealized_gain(asset, price)
                    if gain <= 0 or -desired_change > 2 * band_dollars:
                        plan.long_sells.append((asset, sell_shares))
        else:
            # Want a net short position. Sell out of any long entirely.
            sell = longs.shares_of(asset)
            if sell > 0:
                plan.long_sells.append((asset, sell))
            if harvest_short > 0:
                plan.short_covers.append((asset, harvest_short))
            desired_short = -target_dollars
            have_short = (shorts.shares_of(asset) - harvest_short) * price
            gap = desired_short - have_short
            if gap > band_dollars:
                if not wash.is_blocked("short", asset, step):
                    plan.short_opens.append((asset, gap / price))
            elif gap < -band_dollars:
                cover_shares = min(-gap / price, shorts.shares_of(asset) - harvest_short)
                if cover_shares > 1e-9:
                    gain = shorts.unrealized_gain(asset, price)
                    if gain <= 0 or -gap > 2 * band_dollars:
                        plan.short_covers.append((asset, cover_shares))
    return plan


def execute_plan(
    plan: TradePlan,
    prices: np.ndarray,
    longs: LotBook,
    shorts: LotBook,
    wash: WashBlocker,
    step: int,
) -> tuple[list[Realized], float]:
    """Execute a plan against the ledger. Returns (realized trades, traded $)."""
    realized: list[Realized] = []
    traded = 0.0
    for asset, shares in plan.long_sells:
        recs = longs.close(asset, shares, prices[asset], step)
        realized.extend(recs)
        traded += shares * prices[asset]
        for rec in recs:
            if rec.gain < 0:
                wash.record_loss_sale("long", asset, step)
    for asset, shares in plan.short_covers:
        recs = shorts.close(asset, shares, prices[asset], step)
        realized.extend(recs)
        traded += shares * prices[asset]
        for rec in recs:
            if rec.gain < 0:
                wash.record_loss_sale("short", asset, step)
    for asset, shares in plan.buys:
        longs.open(asset, shares, prices[asset], step)
        traded += shares * prices[asset]
    for asset, shares in plan.short_opens:
        shorts.open(asset, shares, prices[asset], step)
        traded += shares * prices[asset]
    return realized, traded
