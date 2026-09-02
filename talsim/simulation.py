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
    extension_scale: float  # effective L/S extension vs configured (1.0 = full)
    insolvent: bool
    termination_step: int  # last simulated step (n_steps - 1 unless insolvent)
    avg_long_exposure: float
    avg_short_exposure: float
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
    insolvent = False
    terminal_step = cfg.n_steps - 1
    deleverage_events = 0
    max_exposure_error = 0.0
    long_exposures: list[float] = []
    short_exposures: list[float] = []

    def nav_now() -> float:
        return state.cash + ledger.longs.market_value(prices) - ledger.shorts.market_value(prices)

    # Feasibility at inception under the maintenance floors.
    inception_req = (
        cfg.long_maintenance * cfg.long_exposure + cfg.short_maintenance * cfg.short_exposure
    )
    feasible_at_inception = inception_req <= 1.0

    # Feasibility scaling preserves NET exposure: an infeasible book keeps
    # its long-only core and shrinks the long/short extension equally, so
    # every book in a sweep still compares at the same market exposure
    # (250/150 at FINRA floors becomes roughly 233/133, not 228/137).
    net_exposure = cfg.long_exposure - cfg.short_exposure
    extension = cfg.short_exposure
    if not feasible_at_inception and cfg.margin_response == "deleverage":
        ext_max = (cfg.deleverage_buffer - cfg.long_maintenance * net_exposure) / (
            cfg.long_maintenance + cfg.short_maintenance
        )
        extension = max(min(extension, ext_max), 0.0)
    eff_long = net_exposure + extension
    eff_short = extension
    extension_scale = extension / cfg.short_exposure if cfg.short_exposure > 0 else 1.0

    # Alpha calibration at inception (signal-proportional drift).
    weights = target_weights(path.signals[0], cfg.long_exposure, cfg.short_exposure)
    alpha_k = 0.0
    if cfg.alpha_annual:
        target_step = cfg.alpha_annual * (cfg.active_gross / cfg.alpha_reference_active_gross) * dt
        denom = float(np.sum((weights - 1.0 / n) * path.signals[0]))
        if abs(denom) > 1e-9:
            alpha_k = target_step / denom

    # Initial build (at the feasible extension).
    build_weights = target_weights(path.signals[0], eff_long, eff_short)
    plan = plan_trades(cfg, build_weights, cfg.starting_capital, prices, ledger, step=-1)
    _, traded0, cash_delta = execute_plan(plan, prices, ledger, step=-1)
    state.cash += cash_delta
    cost0 = traded0 * cfg.transaction_cost
    state.cash -= cost0
    txn += cost0

    peak = cfg.starting_capital
    max_dd = 0.0
    prev_nav = cfg.starting_capital

    def close_sides(long_fraction: float, short_fraction: float, step: int) -> tuple[float, float]:
        """Force-close fractions of each side. Returns (traded $, txn cost)."""
        traded = 0.0
        for asset in range(n):
            held_long = ledger.longs.shares_of(asset)
            if held_long > 1e-9 and long_fraction > 0:
                take = held_long * long_fraction
                ledger.close("long", asset, take, prices[asset], step)
                state.cash += take * prices[asset]
                traded += take * prices[asset]
            held_short = ledger.shorts.shares_of(asset)
            if held_short > 1e-9 and short_fraction > 0:
                take = held_short * short_fraction
                ledger.close("short", asset, take, prices[asset], step)
                state.cash -= take * prices[asset]
                traded += take * prices[asset]
        cost = traded * cfg.transaction_cost
        state.cash -= cost
        return traded, cost

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
            if ledger.held_days(lot, t) >= 61.0
        )
        qual_div = qualified_mv * cfg.dividend_yield * dt
        ord_div = max(long_mv - qualified_mv, 0.0) * cfg.dividend_yield * dt
        fee_cost = max(nav, 0.0) * cfg.management_fee * dt
        borrow_cost = short_mv * cfg.borrow_cost * dt
        pil_step = 0.0
        for asset, lots in ledger.shorts.lots.items():
            for lot in lots:
                if lot.shares > 1e-9:
                    amount = lot.shares * prices[asset] * cfg.dividend_yield * dt
                    lot.pil_accrued += amount
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
        weights = target_weights(path.signals[t], eff_long, eff_short)
        plan = plan_trades(cfg, weights, max(nav, 1.0), prices, ledger, step=t)
        _, traded, cash_delta = execute_plan(plan, prices, ledger, step=t)
        state.cash += cash_delta
        cost = traded * cfg.transaction_cost
        state.cash -= cost
        txn += cost
        rebalance_traded += traded

        # 4. Exposure audit (invariant, recorded not raised: deferral and
        # wash blocks legitimately bend exposure inside known bounds).
        nav = nav_now()
        if nav > 0:
            long_exp_now = ledger.longs.market_value(prices) / nav
            short_exp_now = ledger.shorts.market_value(prices) / nav
            long_exposures.append(long_exp_now)
            short_exposures.append(short_exp_now)
            net_exp = long_exp_now - short_exp_now
            max_exposure_error = max(max_exposure_error, abs(net_exp - net_exposure))

        # 5. Margin maintenance: respond, don't just observe. A breach is
        # cured by trading back to the compliant target fractions (which
        # preserves net exposure), not by scaling both sides blindly.
        for _ in range(4):
            long_mv = ledger.longs.market_value(prices)
            short_mv = ledger.shorts.market_value(prices)
            nav = nav_now()
            requirement = cfg.long_maintenance * long_mv + cfg.short_maintenance * short_mv
            if nav <= 0:
                min_margin_ratio = min(min_margin_ratio, -1.0)
                deficiency_observed = True
                insolvent = True
                break
            ratio = (nav - requirement) / nav
            min_margin_ratio = min(min_margin_ratio, ratio)
            if ratio >= 0:
                break
            deficiency_observed = True
            if cfg.margin_response == "flag" or requirement <= 0:
                break
            deleverage_events += 1
            target_long_mv = eff_long * nav
            target_short_mv = eff_short * nav
            long_frac = max(0.0, 1 - target_long_mv / long_mv) if long_mv > 0 else 0.0
            short_frac = max(0.0, 1 - target_short_mv / short_mv) if short_mv > 0 else 0.0
            if long_frac < 0.01 and short_frac < 0.01:
                break
            traded_forced, cost_forced = close_sides(long_frac, short_frac, t)
            txn += cost_forced
            if traded_forced <= 1e-6:
                deleverage_events -= 1  # nothing traded; not a real response
                break
        if cfg.margin_response == "deleverage" and not insolvent:
            nav = nav_now()
            if nav > 0:
                long_mv = ledger.longs.market_value(prices)
                short_mv = ledger.shorts.market_value(prices)
                requirement = cfg.long_maintenance * long_mv + cfg.short_maintenance * short_mv
                # Small residuals below the 1%-of-side trade threshold are
                # tolerated; anything larger means the response failed,
                # which is unreachable for configs the validator accepts.
                if nav < requirement - 0.02 * nav:
                    raise AssertionError(
                        "deleverage response ended non-compliant; the net core is infeasible"
                    )
        if insolvent:
            terminal_step = t
            break

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
    # Full liquidation AT the terminal step: the last simulated step
    # normally, or the insolvency step when the path ended early. The
    # unwind shares that step's timestamp, so inception lots on an
    # exactly-one-year horizon have been held exactly 365 days (short
    # term) rather than one extra period. Liquidation records are
    # identified by position in the ledger, not by step, because the
    # final rebalance shares the same step. Settlement uses the year the
    # terminal step actually belongs to.
    # ------------------------------------------------------------------
    final_year = terminal_step // cfg.steps_per_year
    fy0 = final_year * cfg.steps_per_year
    pre_liq_st, pre_liq_lt = ledger.realized_totals(fy0, terminal_step)
    gross_losses_pre_liq = ledger.gross_losses(-1, terminal_step)
    net_realized_pre_liq += pre_liq_st + pre_liq_lt

    liq_step = terminal_step
    n_records_pre_liq = len(ledger.realized)
    liq_traded, liq_cost = close_sides(1.0, 1.0, liq_step)
    txn += liq_cost
    if not ledger.is_empty():
        raise AssertionError("ledger not empty after full liquidation")
    liq_records = ledger.realized[n_records_pre_liq:]

    st_final, lt_final = ledger.realized_totals(fy0, liq_step)
    gross_losses_liq = sum(-rec.gain for rec in liq_records if rec.gain < 0)
    outside = cfg.outside_st_gain_for_year(final_year)
    # Earlier full years were already settled inside the loop; on an early
    # termination nothing after final_year exists to settle.

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
        disallowed_wash_losses=ledger.disallowed_losses(-1, liq_step),
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
        extension_scale=extension_scale,
        insolvent=insolvent,
        termination_step=terminal_step,
        avg_long_exposure=float(np.mean(long_exposures)) if long_exposures else 0.0,
        avg_short_exposure=float(np.mean(short_exposures)) if short_exposures else 0.0,
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


def _run_task(task: tuple[ScenarioConfig, int]) -> PathResult:
    """Module-level so it pickles under the spawn start method."""
    cfg, seed = task
    return run_path(cfg, seed=seed)


def run_sweep(
    cfg: ScenarioConfig,
    books: list[str],
    n_paths: int,
    base_seed: int = 7,
    n_jobs: int | None = 1,
) -> list[SweepResult]:
    """Run each book over common random numbers (path p reuses seed base+p).

    `n_jobs` > 1 runs paths in a process pool (`None` uses every core).
    Each path is a pure function of its config and seed, so parallel runs
    return exactly the serial results in the same order; only wall time
    changes.
    """
    if n_jobs is not None and n_jobs < 1:
        raise ValueError("n_jobs must be >= 1 or None")
    book_cfgs = [cfg.with_book(book) for book in books]
    tasks = [(book_cfg, base_seed + p) for book_cfg in book_cfgs for p in range(n_paths)]
    if n_jobs == 1:
        paths = [_run_task(task) for task in tasks]
    else:
        import os
        from concurrent.futures import ProcessPoolExecutor

        workers = n_jobs or os.cpu_count() or 1
        chunksize = max(1, len(tasks) // (4 * workers))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            paths = list(pool.map(_run_task, tasks, chunksize=chunksize))
    results = []
    for i, (book, book_cfg) in enumerate(zip(books, book_cfgs, strict=True)):
        sweep = SweepResult(book=book, gross_exposure=book_cfg.gross_exposure)
        sweep.paths.extend(paths[i * n_paths : (i + 1) * n_paths])
        results.append(sweep)
    return results
