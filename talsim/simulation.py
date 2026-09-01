"""Lifecycle simulation: rebalance, harvest, pay costs, settle taxes, liquidate.

Wealth accounting, stated once so every output is interpretable:

- The portfolio pays its own way: management fees, borrow, transaction
  costs, payments in lieu, dividend taxes, debit interest on negative cash,
  and any capital-gain tax beyond the household's no-portfolio baseline all
  come out of NAV.
- Tax savings the portfolio creates for the household (losses offsetting
  outside gains, plus the ordinary-income offset) accrue to a side account
  that earns nothing. Ending after-tax wealth = liquidated NAV + side
  account. Crediting savings without growth is deliberate and conservative.
- Dividends on longs are cash income, split into qualified (lots held at
  least one step, a proxy for the 61-day requirement) and non-qualified
  buckets, taxed annually at the preferential and ordinary rates
  respectively. They are ordinary income: capital losses never net against
  them beyond the statutory ordinary-income offset.
- Payments in lieu on shorts are paid in cash as they accrue and added to
  the basis of the shares used to close the short (the Pub 550 treatment
  for shorts held 45 days or less), which reduces the taxable gain on the
  cover rather than vanishing.
- Wash sales are enforced by the ledger itself: disallowed losses move
  into replacement basis with holding-period tacking, whatever the policy
  layer does. The policy also avoids harvesting into a wash and waits out
  the window before re-entering.
- Alpha, when configured, enters as signal-proportional return drift
  calibrated at inception so the initial book's expected active return
  matches the configured rate; realized alpha then varies with how well
  later weights align with later signals. It defaults to zero so leverage
  comparisons stay tax studies.
- Margin is a strategy-level maintenance test at FINRA Rule 4210 floor
  levels. Under the default response, a deficiency triggers forced
  proportional deleveraging through the ledger, with transaction costs and
  realized taxes; the alternative "flag" mode only records the deficiency
  and its results should never be described as implementable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ScenarioConfig
from .lots import Ledger
from .market import generate_path
from .optimize import execute_plan, plan_trades, target_weights
from .tax import settle_year

RETURN_FLOOR = -0.90  # a single step cannot lose more than 90%


@dataclass
class PathResult:
    ending_after_tax_wealth: float
    after_tax_cagr: float
    gross_losses_realized: float  # deductible losses realized BEFORE liquidation
    gross_losses_liquidation: float  # deductible losses realized at liquidation
    disallowed_wash_losses: float  # losses the ledger disallowed (whole path)
    net_realized_pre_liquidation: float  # signed cumulative realized, pre-liquidation
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
    dividend_taxes: float
    debit_interest: float
    min_margin_excess_ratio: float
    maintenance_deficiency_observed: bool
    feasible_at_inception: bool
    deleverage_events: int
    final_exposure_scale: float
    max_net_exposure_error: float
    yearly_taxes_paid: float


@dataclass
class _State:
    cash: float = 0.0


def run_path(cfg: ScenarioConfig, seed: int) -> PathResult:
    path = generate_path(cfg, seed)
    dt = 1.0 / cfg.steps_per_year
    n = cfg.n_assets

    prices = np.full(n, 100.0)
    ledger = Ledger(cfg.steps_per_year, cfg.wash_window_days)
    state = _State(cfg.starting_capital)
    side_account = 0.0
    carry_st = carry_lt = 0.0
    year_qual_div = year_ord_div = 0.0
    pil_accrual: dict[int, float] = {}

    benefit_used_total = 0.0
    taxes_paid_total = 0.0
    dividend_tax_total = 0.0
    net_realized_pre_liq = 0.0
    fees = borrow = txn = pil = debit = 0.0
    rebalance_traded = 0.0
    nav_series: list[float] = []
    active_returns: list[float] = []
    min_margin_ratio = np.inf
    deficiency_observed = False
    deleverage_events = 0
    max_exposure_error = 0.0

    def nav_now() -> float:
        return state.cash + ledger.longs.market_value(prices) - ledger.shorts.market_value(prices)

    def short_extra_basis() -> dict[int, float]:
        out: dict[int, float] = {}
        for asset, accrued in pil_accrual.items():
            held = ledger.shorts.shares_of(asset)
            if held > 1e-9 and accrued > 0:
                out[asset] = accrued / held
        return out

    def consume_pil(asset: int, shares_closed: float, pre_shares: float) -> None:
        if asset in pil_accrual and pre_shares > 1e-9:
            pil_accrual[asset] *= max(0.0, 1 - shares_closed / pre_shares)

    # Feasibility at inception under the maintenance floors.
    inception_req = (
        cfg.long_maintenance * cfg.long_exposure + cfg.short_maintenance * cfg.short_exposure
    )
    feasible_at_inception = inception_req <= 1.0

    # Exposure scale: forced deleveraging shrinks it permanently. A book
    # that re-levered to an infeasible target every quarter would thrash
    # between breach and forced sale on impossible capital.
    exposure_scale = 1.0
    if not feasible_at_inception and cfg.margin_response == "deleverage":
        exposure_scale = cfg.deleverage_buffer / inception_req

    # Alpha calibration at inception (signal-proportional drift).
    weights = target_weights(path.signals[0], cfg.long_exposure, cfg.short_exposure)
    alpha_k = 0.0
    if cfg.alpha_annual:
        target_step = cfg.alpha_annual * (cfg.active_gross / cfg.alpha_reference_active_gross) * dt
        denom = float(np.sum((weights - 1.0 / n) * path.signals[0]))
        if abs(denom) > 1e-9:
            alpha_k = target_step / denom

    # Initial build (at the feasible scale).
    build_weights = target_weights(
        path.signals[0],
        cfg.long_exposure * exposure_scale,
        cfg.short_exposure * exposure_scale,
    )
    plan = plan_trades(cfg, build_weights, cfg.starting_capital, prices, ledger, step=0)
    _, traded0, cash_delta = execute_plan(plan, prices, ledger, step=0)
    state.cash += cash_delta
    cost0 = traded0 * cfg.transaction_cost
    state.cash -= cost0
    txn += cost0

    peak = cfg.starting_capital
    max_dd = 0.0
    prev_nav = cfg.starting_capital

    def close_fraction(fraction: float, step: int) -> float:
        """Force-close a fraction of every position. Returns dollars traded."""
        traded = 0.0
        extra = short_extra_basis()
        for asset in range(n):
            held_long = ledger.longs.shares_of(asset)
            if held_long > 1e-9:
                take = held_long * fraction
                ledger.close("long", asset, take, prices[asset], step)
                state.cash += take * prices[asset]
                traded += take * prices[asset]
            held_short = ledger.shorts.shares_of(asset)
            if held_short > 1e-9:
                take = held_short * fraction
                ledger.close(
                    "short",
                    asset,
                    take,
                    prices[asset],
                    step,
                    extra_basis_per_share=extra.get(asset, 0.0),
                )
                state.cash -= take * prices[asset]
                traded += take * prices[asset]
                consume_pil(asset, take, held_short)
        cost = traded * cfg.transaction_cost
        state.cash -= cost
        nonlocal_txn = cost
        return traded, nonlocal_txn

    for t in range(cfg.n_steps):
        # 1. Market moves.
        step_returns = path.returns[t].copy()
        if alpha_k:
            step_returns = step_returns + alpha_k * path.signals[t]
        step_returns = np.maximum(step_returns, RETURN_FLOOR)
        prices = prices * (1 + step_returns)
        bench_return = float(np.mean(step_returns))

        long_mv = ledger.longs.market_value(prices)
        short_mv = ledger.shorts.market_value(prices)
        nav = nav_now()

        # 2. Cash flows: dividends in (bucketed), costs out.
        qualified_mv = sum(
            lot.shares * prices[asset]
            for asset, lots in ledger.longs.lots.items()
            for lot in lots
            if t - lot.open_step >= 1
        )
        qual_div = qualified_mv * cfg.dividend_yield * dt
        ord_div = max(long_mv - qualified_mv, 0.0) * cfg.dividend_yield * dt
        fee_cost = max(nav, 0.0) * cfg.management_fee * dt
        borrow_cost = short_mv * cfg.borrow_cost * dt
        pil_step = 0.0
        for asset in range(n):
            held = ledger.shorts.shares_of(asset)
            if held > 1e-9:
                amount = held * prices[asset] * cfg.dividend_yield * dt
                pil_accrual[asset] = pil_accrual.get(asset, 0.0) + amount
                pil_step += amount
        state.cash += qual_div + ord_div - fee_cost - borrow_cost - pil_step
        if state.cash > 0:
            state.cash += state.cash * cfg.cash_rate * dt
        elif state.cash < 0:
            debit_cost = -state.cash * cfg.debit_rate * dt
            state.cash -= debit_cost
            debit += debit_cost
        year_qual_div += qual_div
        year_ord_div += ord_div
        fees += fee_cost
        borrow += borrow_cost
        pil += pil_step

        # 3. Rebalance and harvest.
        nav = nav_now()
        weights = target_weights(
            path.signals[t],
            cfg.long_exposure * exposure_scale,
            cfg.short_exposure * exposure_scale,
        )
        plan = plan_trades(cfg, weights, max(nav, 1.0), prices, ledger, step=t)
        pre_short = {a: ledger.shorts.shares_of(a) for a, _ in plan.short_covers}
        extra = short_extra_basis()
        _, traded, cash_delta = execute_plan(plan, prices, ledger, step=t, short_extra_basis=extra)
        for asset, shares in plan.short_covers:
            consume_pil(asset, shares, pre_short[asset])
        state.cash += cash_delta
        cost = traded * cfg.transaction_cost
        state.cash -= cost
        txn += cost
        rebalance_traded += traded

        # 4. Exposure audit (invariant, recorded not raised: deferral and
        # wash blocks legitimately bend exposure inside known bounds).
        nav = nav_now()
        if nav > 0:
            net_exp = (ledger.longs.market_value(prices) - ledger.shorts.market_value(prices)) / nav
            scaled_net = (cfg.long_exposure - cfg.short_exposure) * exposure_scale
            max_exposure_error = max(max_exposure_error, abs(net_exp - scaled_net))

        # 5. Margin maintenance: respond, don't just observe.
        for _ in range(4):
            long_mv = ledger.longs.market_value(prices)
            short_mv = ledger.shorts.market_value(prices)
            nav = nav_now()
            requirement = cfg.long_maintenance * long_mv + cfg.short_maintenance * short_mv
            if nav <= 0:
                min_margin_ratio = min(min_margin_ratio, -1.0)
                deficiency_observed = True
                break
            ratio = (nav - requirement) / nav
            min_margin_ratio = min(min_margin_ratio, ratio)
            if ratio >= 0:
                break
            deficiency_observed = True
            if cfg.margin_response == "flag" or requirement <= 0:
                break
            deleverage_events += 1
            fraction = 1.0 - cfg.deleverage_buffer * nav / requirement
            fraction = min(max(fraction, 0.02), 0.95)
            traded_forced, cost_forced = close_fraction(fraction, t)
            txn += cost_forced
            exposure_scale *= 1.0 - fraction

        # 6. NAV bookkeeping.
        nav = nav_now()
        nav_series.append(nav)
        peak = max(peak, nav)
        max_dd = min(max_dd, nav / peak - 1.0)
        if prev_nav > 0:
            active_returns.append(nav / prev_nav - 1.0 - bench_return)
        prev_nav = nav

        # 7. Year-end settlement (final year settles after liquidation below).
        is_year_end = (t + 1) % cfg.steps_per_year == 0
        is_final = t == cfg.n_steps - 1
        if is_year_end and not is_final:
            year = (t + 1) // cfg.steps_per_year - 1
            y0, y1 = year * cfg.steps_per_year, (year + 1) * cfg.steps_per_year - 1
            st, lt = ledger.realized_totals(y0, y1)
            outside = cfg.outside_st_gain_for_year(year)
            result = settle_year(
                st,
                lt,
                outside,
                carry_st,
                carry_lt,
                cfg.st_rate,
                cfg.lt_rate,
                cfg.ordinary_rate,
                cfg.ordinary_offset_limit,
                qualified_dividends=year_qual_div,
                ordinary_dividends=year_ord_div,
            )
            baseline = outside * cfg.st_rate
            portfolio_tax = max(result.capital_tax - baseline, 0.0) + result.dividend_tax
            state.cash -= portfolio_tax
            taxes_paid_total += portfolio_tax
            dividend_tax_total += result.dividend_tax
            side_account += result.benefit_used
            benefit_used_total += result.benefit_used
            net_realized_pre_liq += st + lt
            carry_st, carry_lt = result.carry_st, result.carry_lt
            year_qual_div = year_ord_div = 0.0

    # ------------------------------------------------------------------
    # Full liquidation at the horizon.
    # ------------------------------------------------------------------
    final_year = cfg.years - 1
    fy0 = final_year * cfg.steps_per_year
    pre_liq_st, pre_liq_lt = ledger.realized_totals(fy0, cfg.n_steps - 1)
    gross_losses_pre_liq = ledger.gross_losses(0, cfg.n_steps - 1)
    net_realized_pre_liq += pre_liq_st + pre_liq_lt

    liq_step = cfg.n_steps
    liq_traded, liq_cost = close_fraction(1.0, liq_step)
    txn += liq_cost
    if not ledger.is_empty():
        raise AssertionError("ledger not empty after full liquidation")

    st_final, lt_final = ledger.realized_totals(fy0, liq_step)
    gross_losses_liq = ledger.gross_losses(liq_step, liq_step)
    outside = cfg.outside_st_gain_for_year(final_year)

    final = settle_year(
        st_final,
        lt_final,
        outside,
        carry_st,
        carry_lt,
        cfg.st_rate,
        cfg.lt_rate,
        cfg.ordinary_rate,
        cfg.ordinary_offset_limit,
        qualified_dividends=year_qual_div,
        ordinary_dividends=year_ord_div,
    )
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
        qualified_dividends=year_qual_div,
        ordinary_dividends=year_ord_div,
    )
    baseline = outside * cfg.st_rate
    final_portfolio_tax = max(final.capital_tax - baseline, 0.0) + final.dividend_tax
    state.cash -= final_portfolio_tax
    taxes_paid_total += final_portfolio_tax
    dividend_tax_total += final.dividend_tax
    side_account += final.benefit_used
    benefit_used_total += final.benefit_used

    # Incremental household tax caused by liquidating, comparing actual
    # after-offset tax bills directly (dividends cancel across the pair).
    tax_with = final.household_tax - final.ordinary_offset * cfg.ordinary_rate
    tax_without = counterfactual.household_tax - counterfactual.ordinary_offset * cfg.ordinary_rate
    liquidation_tax = max(tax_with - tax_without, 0.0)

    ending = state.cash + side_account
    cagr = (ending / cfg.starting_capital) ** (1 / cfg.years) - 1 if ending > 0 else -1.0
    te = (
        float(np.std(active_returns, ddof=1) * np.sqrt(cfg.steps_per_year))
        if len(active_returns) > 2
        else 0.0
    )
    avg_nav = float(np.mean(nav_series)) if nav_series else cfg.starting_capital
    # One-sided turnover, excluding initial construction and liquidation.
    annual_turnover = (rebalance_traded / 2.0) / max(avg_nav, 1.0) / cfg.years

    return PathResult(
        ending_after_tax_wealth=ending,
        after_tax_cagr=cagr,
        gross_losses_realized=gross_losses_pre_liq,
        gross_losses_liquidation=gross_losses_liq,
        disallowed_wash_losses=ledger.disallowed_losses(0, liq_step),
        net_realized_pre_liquidation=net_realized_pre_liq,
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
        dividend_taxes=dividend_tax_total,
        debit_interest=debit,
        min_margin_excess_ratio=float(min_margin_ratio),
        maintenance_deficiency_observed=deficiency_observed,
        feasible_at_inception=feasible_at_inception,
        deleverage_events=deleverage_events,
        final_exposure_scale=exposure_scale,
        max_net_exposure_error=max_exposure_error,
        yearly_taxes_paid=taxes_paid_total,
    )


@dataclass
class SweepResult:
    book: str
    gross_exposure: float
    paths: list[PathResult] = field(default_factory=list)

    def percentile(self, attr: str, q: float) -> float:
        values = [float(getattr(p, attr)) for p in self.paths]
        return float(np.percentile(values, q))

    def median(self, attr: str) -> float:
        return self.percentile(attr, 50)

    def deficiency_probability(self) -> float:
        return float(np.mean([p.maintenance_deficiency_observed for p in self.paths]))


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
