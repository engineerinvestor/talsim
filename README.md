# talsim

[![CI](https://github.com/engineerinvestor/talsim/actions/workflows/ci.yml/badge.svg)](https://github.com/engineerinvestor/talsim/actions/workflows/ci.yml)
[![Docs](https://github.com/engineerinvestor/talsim/actions/workflows/docs.yml/badge.svg)](https://engineerinvestor.github.io/talsim/)
[![PyPI](https://img.shields.io/pypi/v/pytalsim.svg)](https://pypi.org/project/pytalsim/)
[![Python](https://img.shields.io/pypi/pyversions/pytalsim.svg)](https://pypi.org/project/pytalsim/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/engineerinvestor/talsim/blob/master/LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engineerinvestor/talsim/blob/master/examples/talsim_tutorial.ipynb)

A research simulator for **tax-aware long-short (TALS)** portfolio strategies: lot-level tax accounting with enforced wash sales, long/short financing costs, leverage, margin response, full liquidation, and Monte Carlo outcome distributions on a synthetic market.

The question it exists to answer: **when does additional long-short leverage create usable after-tax value, and when does it merely create more turnover, risk, cost, and deferred tax?**

> **Status: v0.4.0, experimental research software.** The engine is synthetic
> and its tax accounting is a documented approximation. Results are
> conditional on stated assumptions and are not evidence about any real
> strategy. Do not use this for personal financial decisions.

## Results at a glance

The headline experiment: five books from long-only to 250/150 traded on the
same 200 simulated market paths, zero manager alpha, $1M for 10 years, full
liquidation at the end. Leverage multiplies harvested losses and still loses
the race after netting, costs, risk, and the terminal tax bill:

![Leverage sweep: losses grow, wealth falls, costs and risk compound](https://raw.githubusercontent.com/engineerinvestor/talsim/master/docs/leverage_sweep.png)

| Book | Median after-tax wealth | Paired diff vs 100/0 | Paths beating 100/0 | Gross losses | Tax benefit used |
|---|---:|---:|---:|---:|---:|
| 100/0 | $1.62M | — | — | $0.77M | $111k |
| 130/30 | $1.50M | −$118k | 29% | $2.43M | $157k |
| 150/50 | $1.40M | −$165k | 26% | $3.15M | $185k |
| 200/100 | $1.26M | −$324k | 26% | $4.70M | $235k |
| 250/150 | $1.13M | −$427k | 19% | $5.54M | $263k |

Medians across 200 common-random-number paths, seed 7 (250/150 is
infeasible at FINRA percentage floors and runs net-preserving at roughly
233/133). 7.2x the gross losses buy 2.4x the usable tax benefit. Every number regenerates from
`python -m talsim.cli sweep --paths 200 --seed 7` on the same platform; the
summary, path-level results, and manifest behind this table are committed
under [`docs/results/`](https://github.com/engineerinvestor/talsim/tree/master/docs/results) and regenerated in pinned CI, and the
figure rebuilds with
`python examples/make_readme_figure.py docs/results/leverage_sweep.csv`.
The 200-path probabilities are demonstration-scale, not inferential
evidence; paired p10/p90 ranges ship in the summary CSV.
These are synthetic research results conditional on stated assumptions, not
evidence about any real strategy.

## What it is

- A deterministic research engine: same config + seed + environment = same result. Floating-point behavior varies across platforms and BLAS builds and can cross discrete trade thresholds, so official artifacts are generated only in pinned CI (runner image, CPython patch version, and numeric stack fixed in `.github/workflows/artifacts.yml` and `requirements-artifacts.txt`), and every manifest records the commit, worktree state, source-tree hash, platform, and full installed-package list that produced it.
- An accounting-first design: the `Ledger` is independent of the trading policy and enforces wash-sale disallowance itself, so any trade list, compliant or not, is accounted correctly.
- Zero-alpha by default. With any positive alpha assumption a leverage comparison silently becomes an alpha study; here alpha is an explicit input, defaulted to zero.

## What it is not

- Not a tax-return calculator. Rules are simplified federal approximations (see below).
- Not an execution or advice system. It never touches real accounts, holdings, or personal data.
- Not empirical validation. The market is synthetic; results are conditional on the configured process.

## Install

```bash
pip install pytalsim            # import talsim; CLI: talsim
pip install "pytalsim[plot]"    # adds matplotlib for the report charts
```

The distribution is named `pytalsim` because PyPI rejects `talsim` as too
similar to an unrelated existing project; the import name and the command
are still `talsim`.

For development, from a clone:

```bash
pip install -e ".[dev]"
pytest            # 56 tests: unit, regression, and property-based (hypothesis)
```

## Quick start

```python
from talsim import ScenarioConfig, run_sweep

cfg = ScenarioConfig()  # $1M, 10y, quarterly, zero alpha, top 2026 federal rates
sweeps = run_sweep(cfg, ["100/0", "130/30"], n_paths=50)
for s in sweeps:
    print(
        s.book,
        f"median wealth ${s.median('ending_after_tax_wealth'):,.0f}",
        f"gross losses ${s.median('gross_losses_realized'):,.0f}",
        f"benefit used ${s.median('tax_benefit_used'):,.0f}",
    )
# 100/0  median wealth $1,722,303 gross losses $751,035 benefit used $109,473
# 130/30 median wealth $1,585,744 gross losses $2,562,042 benefit used $157,310
```

Single-path inspection, with every assumption in one config object:

```python
from talsim import ScenarioConfig, run_path

cfg = ScenarioConfig(long_exposure=1.5, short_exposure=0.5, alpha_annual=0.0)
r = run_path(cfg, seed=7)
print(
    f"wealth ${r.ending_after_tax_wealth:,.0f}, TE {r.tracking_error:.1%}, "
    f"turnover {r.annual_turnover:.1f}x, washed ${r.disallowed_wash_losses:,.0f}"
)
# wealth $2,393,151, TE 9.7%, turnover 3.0x, washed $0
```

Or from the command line:

```bash
talsim sweep --paths 200 --seed 7 --out results/
talsim scenarios --paths 100 --seed 7 --out results/
```

(`python -m talsim.cli` is equivalent to the `talsim` command.)

Each run writes a summary CSV, a **path-level CSV** (every path, with its seed, so any statistic can be recomputed), and a manifest recording the package version, git commit, Python and NumPy versions, the full config of every scenario, and SHA-256 checksums of the outputs. The sweep summary includes **paired differences versus 100/0 on common random numbers** (median difference and probability of beating the baseline), which are far more informative than medians alone.

`scripts/bootstrap_intervals.py` reads the path-level CSVs and writes `*_intervals.csv` beside them: a 95% bootstrap interval for each paired median difference and a Wilson interval for each win probability. These are sampling intervals within the model, not model error.

### Summitward export

The interactive calculator in Summitward's [TALS simulator guide](https://summitward.com/learn/tals-leverage-simulator#worth-it) reads a precomputed grid rather than running the engine live:

```bash
python scripts/run_grid.py --paths 100 --seed 7 --workers 12 --out results/
python scripts/gen_summitward_grid.py results/            # writes web/src/lib/talsim-grid.ts
```

`run_grid.py` spans book x outside-gains ratio x cost tier x alpha x horizon x federal bracket at a $1M reference capital (180 cells, 900 book-cells) and writes the same summary, path-level, and manifest files as the CLI. `gen_summitward_ts.py` does the same for the static charts from `sweep` and `scenarios` output. Both exporters refuse to run if a manifest checksum does not match its CSV.

## Tutorial

A short notebook walks through the API end to end: one path, the five
accounting quantities, a leverage sweep on common random numbers, the report
figure, an outside-gain what-if, margin feasibility, and reproducibility. It
runs in about a minute, and CI executes it on every push. Its path counts are
small, so its numbers are illustrative; the official results above come from
pinned CI.

- Open in Colab: https://colab.research.google.com/github/engineerinvestor/talsim/blob/master/examples/talsim_tutorial.ipynb
- Source: https://github.com/engineerinvestor/talsim/blob/master/examples/talsim_tutorial.ipynb

## The accounting the reports keep separate

More harvested losses are not more wealth. Every report distinguishes:

1. **Gross losses realized (pre-liquidation)**: deductible realized losses before the terminal unwind, net of wash disallowance.
2. **Disallowed wash losses**: losses the ledger disallowed; their value moved into replacement basis (with holding-period tacking) rather than vanishing.
3. **Net realized result**: what survives netting against the portfolio's own realized gains.
4. **Tax benefit used**: the household tax actually saved against outside gains plus the $3,000 ordinary offset; the only number that deserves to be called a benefit.
5. **Liquidation tax**: the incremental household tax caused by the terminal unwind, measured against settling the final year without liquidating.

## Model mechanics (v0.4.0)

- **Wash sales are enforced in the ledger**, both directions of the window, share-matched **in acquisition order with lot splitting**: when only part of a replacement lot matches, the matched shares become their own sublot carrying the transferred basis and a tacked TAX holding clock, while their actual acquisition date (which drives the wash window, the PIL 45-day test, and dividend qualification) is preserved separately. Short-side replacements have the deferred loss subtracted from their basis (sale proceeds), never added. **The window is an exact elapsed-day comparison**: at quarterly cadence a same-step repurchase washes and the next quarter, 91 days later, legally does not. Long-term character requires MORE than 365 days, per Pub 550. The policy layer independently avoids washes: it will not harvest a freshly bought name, it waits out the window before re-entering, redistributes blocked exposure to substitute names (capped at 2x each name's own target), and risk-driven reductions of recent buys sell gain lots first.
- **Exposure is constructed from post-trade state per side**, never signed drift, so short-to-long transitions land on target. A harvest floor prevents a side from flattening itself when every position is at a loss at once. Realized net exposure error is recorded per path.
- **Dividends are ordinary income**, split qualified/non-qualified by a day-based holding test (61 days, a proxy for the statutory 60-days-in-121 rule, correct at any cadence), taxed annually in their own buckets; capital losses never absorb them beyond the statutory ordinary offset. **Payments in lieu accrue per short lot** and are capitalized into cover basis only when the short is closed within 45 days (Pub 550); longer-held PIL gets no tax benefit, a deliberate conservatism until an investment-interest bucket exists.
- **Negative cash accrues debit interest** (default 6%); positive cash earns a configurable rate (default zero, deliberately conservative).
- **Margin** is a strategy-level maintenance test at FINRA Rule 4210 percentage floors (25% long / 30% short; the rule's per-share short minima for low-priced stocks are not modeled). Feasibility scaling **preserves net exposure**: an infeasible book keeps its long-only core and shrinks the long/short extension equally, so 250/150 at floor requirements runs as roughly 233/133 (`extension_scale` reports the shrinkage) and every book in a sweep compares at the same market exposure. A deficiency during the path is cured by trading back to the compliant target fractions, with transaction costs and tax consequences; nonpositive equity ends the path in an explicit insolvent state. A "flag" mode records deficiencies without responding; its results should never be described as implementable. Actual average long and short exposures are reported per path.
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
examples/
  talsim_tutorial.ipynb   # end-to-end tutorial (Colab link in the first cell)
  make_readme_figure.py   # rebuilds docs/leverage_sweep.png from the summary CSV
```

## Documentation

API documentation is published from the module docstrings at
**https://engineerinvestor.github.io/talsim/** on every push to master.

## Changelog

**0.4.0** — Third correctness release. The wash-sale window is now an
exact elapsed-day comparison (the previous step-rounded window disallowed
legal 91-day repurchases at quarterly cadence, materially suppressing
harvests and inflating the leverage penalty); actual acquisition, tacked
tax holding, and PIL clocks are separate fields; long-term character
requires more than 365 days; early insolvency liquidates at its actual
step and settles its actual year (with a real regression test replacing a
vacuous one); configurations whose net core is infeasible at maintenance
floors are rejected in deleverage mode; ledger operations validate inputs
before mutating and reject unknown sides; all config values, including
every outside-gain event, must be finite and the offset limit
non-negative; the terminal unwind shares the final step (it was stamped
one step later, granting every lot an extra period of holding time, so an
inception lot on an exactly-one-year horizon counted as long term);
manifests record worktree state, source hash, platform, and full package
versions; official artifacts move to pinned CI; first PyPI release, as
`pytalsim`. Results produced by 0.3.0 should be discarded.

**0.3.0** — Second correctness release following a follow-up external
review. Partial wash-sale matches now SPLIT replacement lots (matched
shares get the basis transfer and tacked holding period; unmatched shares
keep their own), matching walks purchases chronologically instead of the
HIFO-sorted view, and a property-based test suite caught and fixed a
short-side sign error in basis transfer (deferred losses now reduce a
replacement short's basis). Payments in lieu accrue per lot and respect
the 45-day capitalization boundary; dividend qualification and holding
periods are day-based at any cadence; margin feasibility scaling preserves
net exposure (250/150 runs as ~233/133); nonpositive equity is an explicit
insolvency state; configuration and CLI inputs are validated; mypy runs in
CI. Results produced by 0.2.0 should be discarded.

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

## Citation

If you use talsim in academic work, please cite it:

```bibtex
@software{talsim,
  author  = {{Engineer Investor}},
  title   = {talsim: a research simulator for tax-aware long-short
             portfolio strategies},
  year    = {2026},
  version = {0.4.0},
  url     = {https://github.com/engineerinvestor/talsim},
  license = {MIT},
  note    = {Synthetic-market research software; results are conditional
             on configured assumptions}
}
```

A machine-readable [`CITATION.cff`](https://github.com/engineerinvestor/talsim/blob/master/CITATION.cff) is included, so GitHub's
"Cite this repository" button produces the same reference.

## License

MIT. This is educational research software, not tax, legal, accounting, or investment advice.
