# talsim

A research simulator for **tax-aware long-short (TALS)** portfolio strategies: lot-level tax accounting with enforced wash sales, long/short financing costs, leverage, margin response, full liquidation, and Monte Carlo outcome distributions on a synthetic market.

The question it exists to answer: **when does additional long-short leverage create usable after-tax value, and when does it merely create more turnover, risk, cost, and deferred tax?**

> **Status: v0.2.0, experimental research software.** The engine is synthetic
> and its tax accounting is a documented approximation. Results are
> conditional on stated assumptions and are not evidence about any real
> strategy. Do not use this for personal financial decisions.

## What it is

- A deterministic, seed-reproducible research engine. Same config + seed = same result, always.
- An accounting-first design: the `Ledger` is independent of the trading policy and enforces wash-sale disallowance itself, so any trade list, compliant or not, is accounted correctly.
- Zero-alpha by default. With any positive alpha assumption a leverage comparison silently becomes an alpha study; here alpha is an explicit input, defaulted to zero.

## What it is not

- Not a tax-return calculator. Rules are simplified federal approximations (see below).
- Not an execution or advice system. It never touches real accounts, holdings, or personal data.
- Not empirical validation. The market is synthetic; results are conditional on the configured process.

## Install

```bash
pip install -e ".[dev]"
pytest            # 42 tests, including regression tests for past accounting defects
```

## Quick start

```python
from talsim import ScenarioConfig, run_sweep

cfg = ScenarioConfig()  # $1M, 10y, quarterly, zero alpha, top 2026 federal rates
sweeps = run_sweep(cfg, ["100/0", "130/30", "150/50", "200/100", "250/150"], n_paths=200)
for s in sweeps:
    print(s.book, round(s.median("ending_after_tax_wealth")), s.deficiency_probability())
```

Or from the command line:

```bash
python -m talsim.cli sweep --paths 200 --seed 7 --out results/
python -m talsim.cli scenarios --paths 100 --seed 7 --out results/
```

Each run writes a summary CSV, a **path-level CSV** (every path, with its seed, so any statistic can be recomputed), and a manifest recording the package version, git commit, Python and NumPy versions, the full config of every scenario, and SHA-256 checksums of the outputs. The sweep summary includes **paired differences versus 100/0 on common random numbers** (median difference and probability of beating the baseline), which are far more informative than medians alone.

## The accounting the reports keep separate

More harvested losses are not more wealth. Every report distinguishes:

1. **Gross losses realized (pre-liquidation)**: deductible realized losses before the terminal unwind, net of wash disallowance.
2. **Disallowed wash losses**: losses the ledger disallowed; their value moved into replacement basis (with holding-period tacking) rather than vanishing.
3. **Net realized result**: what survives netting against the portfolio's own realized gains.
4. **Tax benefit used**: the household tax actually saved against outside gains plus the $3,000 ordinary offset; the only number that deserves to be called a benefit.
5. **Liquidation tax**: the incremental household tax caused by the terminal unwind, measured against settling the final year without liquidating.

## Model mechanics (v0.2.0)

- **Wash sales are enforced in the ledger**, both directions of the window (purchases before or after a loss sale), share-matched, with basis transfer and holding-period tacking. The window is expressed in steps, always rounded up, so a coarser cadence over-blocks and never under-blocks. The policy layer independently avoids washes: it will not harvest a freshly bought name, it waits out the window before re-entering, it redistributes blocked exposure to substitute names (capped per name), and risk-driven reductions of recent buys sell gain lots first.
- **Exposure is constructed from post-trade state per side**, never signed drift, so short-to-long transitions land on target. A harvest floor prevents a side from flattening itself when every position is at a loss at once. Realized net exposure error is recorded per path.
- **Dividends are ordinary income**, split qualified/non-qualified by a holding-period proxy, taxed annually in their own buckets; capital losses never absorb them beyond the statutory ordinary offset. Payments in lieu on shorts are paid in cash and added to the basis of shares used to close (the Pub 550 treatment for shorts held 45 days or less).
- **Negative cash accrues debit interest** (default 6%); positive cash earns a configurable rate (default zero, deliberately conservative).
- **Margin** is a strategy-level maintenance test at FINRA Rule 4210 floor levels (25% long / 30% short). Under the default response, a deficiency triggers forced proportional deleveraging through the ledger, with transaction costs and real tax consequences, and the book's exposure scale shrinks permanently (no re-levering on impossible capital). A book that is infeasible at inception, like 250/150 under floor requirements, opens at the largest feasible scale and reports it (`final_exposure_scale`). A "flag" mode records deficiencies without responding; its results should never be described as implementable.
- **Alpha**, when configured, enters as signal-proportional return drift calibrated at inception; the equal-weight 100/0 baseline has no active positions and receives none.
- **Tracking error** is measured against an investable equal-weight portfolio of the same universe, and includes cost and tax drag. **Turnover** is one-sided (traded dollars / 2) over average NAV per year, excluding initial construction and terminal liquidation.

## Remaining simplifications (read before citing any number)

- One wash group per (side, asset). Household scope (spouse, IRA, controlled entities), where a washed loss can be permanently destroyed rather than deferred, is out of scope.
- Short-sale gains/losses are treated as short-term; long-term short edge cases are not modeled.
- No delistings, corporate actions, borrow recalls, hard-to-borrow spikes, jumps, volatility clustering, intraperiod margin events, or capacity limits. Returns are Gaussian per step, floored at -90%.
- The trading policy is a transparent heuristic (rank tilts, bands, deferral), not a risk-model-constrained optimizer; `risk.py`'s estimators are provided for analysis and are not wired into construction.
- Federal only, top 2026 rates including NIIT by default; no state tax.
- Tax savings accrue to a zero-return side account rather than compounding.

## Layout

```
talsim/
  config.py       # every assumption, validated; presets 100/0 .. 250/150
  lots.py         # lot ledger, HIFO closes, enforced wash sales, basis transfer
  tax.py          # netting, dividend buckets, $3k offset, carryforwards
  market.py       # synthetic factor market + persistent signal
  risk.py         # sample/EWMA/Ledoit-Wolf/OAS covariance, PSD repair
  optimize.py     # per-side state targets, harvest floor, substitute redistribution
  simulation.py   # lifecycle loop, costs, margin response, liquidation, Monte Carlo
  plotting.py     # report charts (optional matplotlib extra)
  cli.py          # reproducible runs, path-level output, provenance manifests
```

## Changelog

**0.2.0** — Correctness release following external review. Wash-sale
enforcement moved into the ledger (the previous policy-only check allowed
same-step harvest-and-rebuy, overstating harvested losses); trade
construction rebuilt from per-side state (short-to-long transitions
previously overshot and created free leverage, now debit interest accrues);
dividends moved out of the capital-gain buckets (they were nettable against
losses without limit); payments in lieu now adjust cover basis; metric
definitions corrected (pre-liquidation snapshots, direct-comparison
liquidation tax); margin deficiencies now force deleveraging with a
persistent exposure scale. Results produced by 0.1.0 should be discarded.

**0.1.0** — Initial release.

## License

MIT. This is educational research software, not tax, legal, accounting, or investment advice.
