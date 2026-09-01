# talsim

A research simulator for **tax-aware long-short (TALS)** portfolio strategies: lot-level tax accounting, wash-sale re-entry blocking, long/short financing costs, leverage, margin flags, full liquidation, and Monte Carlo outcome distributions on a synthetic market.

The question it exists to answer: **when does additional long-short leverage create usable after-tax value, and when does it merely create more turnover, risk, cost, and deferred tax?**

## What it is

- A deterministic, seed-reproducible research engine. Same config + seed = same result, always.
- An accounting-first design: the lot ledger is independent of the trading policy, so any trade list can be audited by the same books.
- Zero-alpha by default. With any positive alpha assumption, a leverage comparison silently becomes an alpha study; here alpha is an explicit, prominently defaulted-to-zero input.

## What it is not

- Not a tax-return calculator. Tax rules are simplified research approximations (see below).
- Not an execution or advice system. It never touches real accounts, holdings, or personal data.
- Not empirical validation. The market is synthetic; results are conditional on the configured process.

## Install

```bash
pip install -e ".[dev]"
pytest            # 26 tests
```

## Quick start

```python
from talsim import ScenarioConfig, run_sweep

cfg = ScenarioConfig()  # $1M, 10y, quarterly, zero alpha, top 2026 federal rates
sweeps = run_sweep(cfg, ["100/0", "130/30", "150/50", "200/100", "250/150"], n_paths=80)
for s in sweeps:
    print(s.book, round(s.median("ending_after_tax_wealth")), s.margin_call_probability())
```

Or from the command line, writing CSVs plus a manifest of every assumption:

```bash
python -m talsim.cli sweep --paths 80 --seed 7 --out results/
python -m talsim.cli scenarios --paths 40 --seed 7 --out results/
```

## The five quantities the reports keep separate

More harvested losses are not more wealth. Every report distinguishes:

1. **Gross losses realized**: total dollars of realized losses before liquidation.
2. **Net capital loss**: what survives netting against the portfolio's own realized gains.
3. **Tax benefit used**: the reduction in household tax actually achieved (against outside gains and the $3,000 ordinary-income offset), the only number that deserves to be called a benefit.
4. **Unused loss carryforward**: losses banked but never used within the horizon.
5. **Liquidation tax**: the extra tax the final unwind adds, which claws back deferral.

## Model simplifications (read before citing any number)

- One wash-sale group per asset; re-entry after a loss sale is blocked for one step (91 days at quarterly cadence), which **over-blocks** relative to the statute's 30 days: conservative, never generous. Household-scope wash sales (spouse, IRA, controlled entities) are out of scope.
- Short-sale gains/losses are treated as short-term (the common case under Pub 550's delivered-shares rule; long-term edge cases are not modeled).
- Payments in lieu of dividends on shorts are a pure cost (matching shorts held under the 46-day deductibility threshold); dividends on longs are cash taxed annually at the long-term rate.
- The margin check is a strategy-level maintenance test at FINRA Rule 4210 floor levels (25% long / 30% short). It flags paths; it does not force liquidation. It is not Reg T, portfolio margin, or any broker's house policy.
- Tax savings accrue to a zero-return side account rather than compounding: conservative and explicit.
- Synthetic market: market + sector + idiosyncratic factors, persistent AR(1) cross-sectional signal, no delistings, corporate actions, borrow recalls, or capacity limits.
- Federal only (top 2026 rates incl. NIIT by default); no state tax.

## Layout

```
talsim/
  config.py       # every assumption, validated; presets 100/0 .. 250/150
  lots.py         # lot inventory, HIFO closes, wash-sale blocking
  tax.py          # netting, $3k ordinary offset, carryforwards, benefit decomposition
  market.py       # synthetic factor market + persistent signal
  risk.py         # sample/EWMA/Ledoit-Wolf/OAS covariance, PSD repair
  optimize.py     # exposure-exact target weights, tax-aware trade planning
  simulation.py   # lifecycle loop, costs, margin flags, liquidation, Monte Carlo
  plotting.py     # report charts (optional matplotlib extra)
  cli.py          # reproducible runs + manifests
```

## License

MIT. This is educational research software, not tax, legal, accounting, or investment advice.
