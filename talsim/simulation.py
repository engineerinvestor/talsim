"""Lifecycle simulation: rebalance, harvest, pay costs, settle taxes, liquidate.

Wealth accounting, stated once so every output is interpretable:

- The portfolio pays its own way: management fees, borrow, transaction costs,
  and payments in lieu of dividends come out of NAV, as does any tax the
  portfolio's net realized gains add on top of the household's baseline.
- Tax savings the portfolio creates for the household (losses offsetting
  outside gains, plus the ordinary-income offset) accrue to a side account
  that earns nothing. Ending after-tax wealth = liquidated NAV + side
  account. Crediting savings without growth is deliberate and conservative.
- Dividends on longs are received in cash and taxed annually at the
  long-term (qualified) rate; shorts pay the full dividend in lieu, treated
  as a cost rather than a deduction, which matches the common case of shorts
  held under the 46-day deductibility threshold.
- Alpha, when configured, is a deterministic drift added to the active book,
  scaled with active gross exposure. It defaults to zero so that leverage
  comparisons stay tax studies instead of silently becoming alpha studies.

The margin check is a strategy-level maintenance test at FINRA Rule 4210
floor levels (25% long, 30% short). It is not a broker margin engine: no
Reg T initial margin, no portfolio margin, no house requirements, no
per-share minimums, and breaching it does not force liquidation in the
model; paths are flagged instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ScenarioConfig
from .lots import LotBook, WashBlocker
from .market import generate_path
from .optimize import execute_plan, plan_trades, target_weights
from .tax import settle_year


@dataclass
class PathResult:
    ending_after_tax_wealth: float
    after_tax_cagr: float
    gross_losses_realized: float  # pre-liquidation, absolute dollars
    net_loss_pre_liquidation: float  # cumulative net capital loss before final year
    tax_benefit_used: float
    unused_loss_carry: float
    liquidation_tax: float
    tracking_error: float
    max_drawdown: float
    annual_turnover: float
    management_fees: float
    borrow_costs: float
    transaction_costs: float
    payments_in_lieu: float
    min_margin_excess_ratio: float  # min over path of (equity - requirement) / equity
    margin_breached: bool
    yearly_taxes_paid: float


@dataclass
class _Cash:
    balance: float = 0.0


def run_path(cfg: ScenarioConfig, seed: int) -> PathResult:
    path = generate_path(cfg, seed)
    dt = 1.0 / cfg.steps_per_year

    prices = np.full(cfg.n_assets, 100.0)
    longs = LotBook("long", cfg.steps_per_year)
    shorts = LotBook("short", cfg.steps_per_year)
    wash = WashBlocker(block_steps=1)
    cash = _Cash(cfg.starting_capital)
    side_account = 0.0
    carry_st = carry_lt = 0.0
    year_st = year_lt = 0.0

    gross_losses = 0.0
    benefit_used_total = 0.0
    taxes_paid_total = 0.0
    net_loss_pre_liq = 0.0
    fees = borrow = txn = pil = 0.0
    traded_total = 0.0
    nav_series: list[float] = []
    active_returns: list[float] = []
    min_margin_ratio = np.inf
    margin_breached = False

    alpha_step = (
        cfg.alpha_annual * (cfg.active_gross / cfg.alpha_reference_active_gross) * dt
        if cfg.alpha_annual
        else 0.0
    )

    def nav_now() -> float:
        # Short proceeds sit in cash from when each short was opened, so NAV
        # is cash plus long value minus the cost to buy back every short.
        return cash.balance + longs.market_value(prices) - shorts.market_value(prices)

    # Initial build at step 0 targets.
    nav = cfg.starting_capital
    weights = target_weights(path.signals[0], cfg.long_exposure, cfg.short_exposure)
    plan = plan_trades(cfg, weights, nav, prices, longs, shorts, wash, step=0)
    _, traded = execute_plan(plan, prices, longs, shorts, wash, step=0)
    cash.balance = cfg.starting_capital - longs.market_value(prices) + (shorts.market_value(prices))
    traded_total += traded
    txn0 = traded * cfg.transaction_cost
    cash.balance -= txn0
    txn += txn0

    peak = cfg.starting_capital
    max_dd = 0.0

    for t in range(cfg.n_steps):
        # 1. Market moves; short liability moves with prices.
        step_returns = path.returns[t].copy()
        if alpha_step:
            # Drift applied to active positions in proportion to |active weight|.
            active = weights - 1.0 / cfg.n_assets
            gross_active = np.abs(active).sum()
            if gross_active > 0:
                step_returns += alpha_step * active / gross_active * cfg.n_assets
        prices = prices * (1 + step_returns)

        long_mv = longs.market_value(prices)
        short_mv = shorts.market_value(prices)
        nav = nav_now()

        # 2. Cash flows: dividends in, costs out.
        div_cash = long_mv * cfg.dividend_yield * dt
        pil_cost = short_mv * cfg.dividend_yield * dt
        fee_cost = max(nav, 0.0) * cfg.management_fee * dt
        borrow_cost = short_mv * cfg.borrow_cost * dt
        cash.balance += div_cash - pil_cost - fee_cost - borrow_cost
        pil += pil_cost
        fees += fee_cost
        borrow += borrow_cost
        year_lt += div_cash  # qualified dividends taxed at the LT rate annually

        nav = nav_now()

        # 3. Margin check before trading (strategy-level, flag only).
        equity = nav
        requirement = cfg.long_maintenance * long_mv + cfg.short_maintenance * short_mv
        if equity > 0:
            ratio = (equity - requirement) / equity
            min_margin_ratio = min(min_margin_ratio, ratio)
            if ratio < 0:
                margin_breached = True
        else:
            min_margin_ratio = min(min_margin_ratio, -1.0)
            margin_breached = True

        # 4. Rebalance and harvest.
        weights = target_weights(path.signals[t], cfg.long_exposure, cfg.short_exposure)
        plan = plan_trades(cfg, weights, max(nav, 1.0), prices, longs, shorts, wash, step=t)
        realized, traded = execute_plan(plan, prices, longs, shorts, wash, step=t)
        traded_total += traded
        txn_cost = traded * cfg.transaction_cost
        txn += txn_cost
        for rec in realized:
            if rec.gain < 0:
                gross_losses += -rec.gain
            if rec.term == "st":
                year_st += rec.gain
            else:
                year_lt += rec.gain
        # Cash impact of trades: long sells/buys and short opens/covers.
        cash.balance += sum(s * prices[a] for a, s in plan.long_sells)
        cash.balance -= sum(s * prices[a] for a, s in plan.buys)
        cash.balance += sum(s * prices[a] for a, s in plan.short_opens)
        cash.balance -= sum(s * prices[a] for a, s in plan.short_covers)
        cash.balance -= txn_cost

        nav = nav_now()
        nav_series.append(nav)
        peak = max(peak, nav)
        max_dd = min(max_dd, nav / peak - 1.0)
        market_step = path.market_returns[t]
        prev_nav = nav_series[-2] if len(nav_series) > 1 else cfg.starting_capital
        if prev_nav > 0:
            active_returns.append(nav / prev_nav - 1.0 - market_step)

        # 5. Year-end tax settlement.
        is_year_end = (t + 1) % cfg.steps_per_year == 0
        is_final = t == cfg.n_steps - 1
        if is_year_end and not is_final:
            year = (t + 1) // cfg.steps_per_year - 1
            outside = cfg.outside_st_gain_for_year(year)
            result = settle_year(
                year_st,
                year_lt,
                outside,
                carry_st,
                carry_lt,
                cfg.st_rate,
                cfg.lt_rate,
                cfg.ordinary_rate,
                cfg.ordinary_offset_limit,
            )
            baseline = outside * cfg.st_rate
            portfolio_tax = max(result.tax_paid - baseline, 0.0)
            cash.balance -= portfolio_tax
            taxes_paid_total += portfolio_tax
            side_account += result.benefit_used
            benefit_used_total += result.benefit_used
            net_loss_pre_liq += -min(year_st + year_lt, 0.0)
            carry_st, carry_lt = result.carry_st, result.carry_lt
            year_st = year_lt = 0.0

    # 6. Full liquidation at the horizon.
    pre_liq_st, pre_liq_lt = year_st, year_lt
    for asset in range(cfg.n_assets):
        held = longs.shares_of(asset)
        if held > 0:
            for rec in longs.close(asset, held, prices[asset], cfg.n_steps):
                if rec.gain < 0:
                    gross_losses += -rec.gain
                if rec.term == "st":
                    year_st += rec.gain
                else:
                    year_lt += rec.gain
            cash.balance += held * prices[asset]
        short_held = shorts.shares_of(asset)
        if short_held > 0:
            for rec in shorts.close(asset, short_held, prices[asset], cfg.n_steps):
                if rec.gain < 0:
                    gross_losses += -rec.gain
                year_st += rec.gain
            # Cover at current market value; opening proceeds are already in cash.
            cash.balance -= short_held * prices[asset]

    final_year = cfg.years - 1
    outside = cfg.outside_st_gain_for_year(final_year)
    final = settle_year(
        year_st,
        year_lt,
        outside,
        carry_st,
        carry_lt,
        cfg.st_rate,
        cfg.lt_rate,
        cfg.ordinary_rate,
        cfg.ordinary_offset_limit,
    )
    baseline = outside * cfg.st_rate
    final_portfolio_tax = max(final.tax_paid - baseline, 0.0)
    cash.balance -= final_portfolio_tax
    taxes_paid_total += final_portfolio_tax
    side_account += final.benefit_used
    benefit_used_total += final.benefit_used

    # Liquidation tax: the extra tax caused by the final-year realizations,
    # measured against settling the final year without liquidating.
    counterfactual = settle_year(
        pre_liq_st,
        pre_liq_lt,
        outside,
        carry_st,
        carry_lt,
        cfg.st_rate,
        cfg.lt_rate,
        cfg.ordinary_rate,
        cfg.ordinary_offset_limit,
    )
    liquidation_tax = max(
        (final.tax_paid - final.benefit_used)
        - (counterfactual.tax_paid - counterfactual.benefit_used),
        0.0,
    )

    ending = cash.balance + side_account
    years = cfg.years
    cagr = (ending / cfg.starting_capital) ** (1 / years) - 1 if ending > 0 else -1.0
    te = (
        float(np.std(active_returns, ddof=1) * np.sqrt(cfg.steps_per_year))
        if (len(active_returns) > 2)
        else 0.0
    )
    avg_nav = float(np.mean(nav_series)) if nav_series else cfg.starting_capital
    annual_turnover = traded_total / max(avg_nav, 1.0) / cfg.years

    return PathResult(
        ending_after_tax_wealth=ending,
        after_tax_cagr=cagr,
        gross_losses_realized=gross_losses,
        net_loss_pre_liquidation=net_loss_pre_liq,
        tax_benefit_used=benefit_used_total,
        unused_loss_carry=final.carry_st + final.carry_lt,
        liquidation_tax=liquidation_tax,
        tracking_error=te,
        max_drawdown=max_dd,
        annual_turnover=annual_turnover,
        management_fees=fees,
        borrow_costs=borrow,
        transaction_costs=txn,
        payments_in_lieu=pil,
        min_margin_excess_ratio=float(min_margin_ratio),
        margin_breached=margin_breached,
        yearly_taxes_paid=taxes_paid_total,
    )


@dataclass
class SweepResult:
    book: str
    gross_exposure: float
    paths: list[PathResult] = field(default_factory=list)

    def percentile(self, attr: str, q: float) -> float:
        values = [getattr(p, attr) for p in self.paths]
        return float(np.percentile(values, q))

    def median(self, attr: str) -> float:
        return self.percentile(attr, 50)

    def margin_call_probability(self) -> float:
        return float(np.mean([p.margin_breached for p in self.paths]))


def run_sweep(
    cfg: ScenarioConfig, books: list[str], n_paths: int, base_seed: int = 7
) -> list[SweepResult]:
    """Run each book over common random numbers (path p reuses seed base+p)."""
    results = []
    for book in books:
        book_cfg = cfg.with_book(book)
        sweep = SweepResult(book=book, gross_exposure=book_cfg.gross_exposure)
        for p in range(n_paths):
            sweep.paths.append(run_path(book_cfg, seed=base_seed + p))
        results.append(sweep)
    return results
