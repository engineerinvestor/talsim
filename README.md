# talsim

[![CI](https://github.com/engineerinvestor/talsim/actions/workflows/ci.yml/badge.svg)](https://github.com/engineerinvestor/talsim/actions/workflows/ci.yml)
[![Docs](https://github.com/engineerinvestor/talsim/actions/workflows/docs.yml/badge.svg)](https://engineerinvestor.github.io/talsim/)
[![PyPI](https://img.shields.io/pypi/v/pytalsim.svg)](https://pypi.org/project/pytalsim/)
[![Python](https://img.shields.io/pypi/pyversions/pytalsim.svg)](https://pypi.org/project/pytalsim/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/engineerinvestor/talsim/blob/master/LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engineerinvestor/talsim/blob/master/examples/talsim_tutorial.ipynb)

A research simulator for **tax-aware long-short (TALS)** portfolio strategies: lot-level tax accounting with enforced wash sales, long/short financing costs, leverage, margin response, full liquidation, and Monte Carlo outcome distributions on a synthetic market.

The question it exists to answer: **when does additional long-short leverage create usable after-tax value, and when does it merely create more turnover, risk, cost, and deferred tax?**

> **Status: v0.5.0, experimental research software.** The engine is synthetic
> and its tax accounting is a documented approximation. Results are
> conditional on stated assumptions and are not evidence about any real
> strategy. Do not use this for personal financial decisions.

## Try it in your browser

An interactive explorer runs at **https://talsim.streamlit.app**: browse the
pinned-CI official results, or run the actual engine on your own assumptions
(bounded small-sample runs; the tables in this README come from the 200-path
pinned-CI run).

## Results at a glance

The headline experiment: five books from long-only to 250/150 traded on the
same 200 simulated market paths of a 500-name universe, zero manager alpha,
$1M for 10 years, portfolio margin, full liquidation at the end. Leverage
multiplies harvested losses and still loses the race after netting, costs,
risk, and the terminal tax bill:

![Leverage sweep: losses grow, wealth falls, costs and risk compound](https://raw.githubusercontent.com/engineerinvestor/talsim/master/docs/leverage_sweep.png)

| Book | Median after-tax wealth | Paired diff vs 100/0 | Paths beating 100/0 | Gross losses | Tax benefit used |
|---|---:|---:|---:|---:|---:|
| 100/0 | $1.66M | — | — | $0.77M | $120k |
| 130/30 | $1.59M | −$54k | 25% | $2.47M | $146k |
| 150/50 | $1.57M | −$69k | 24% | $3.26M | $167k |
| 200/100 | $1.52M | −$120k | 18% | $5.10M | $215k |
| 250/150 | $1.46M | −$194k | 14% | $6.60M | $257k |

Medians across 200 common-random-number paths, seed 7. 8.6x the gross
losses buy 2.1x the usable tax benefit. The gap is less than half of what
v0.4 reported: the 36-name universe of earlier releases produced tracking
errors of 10 to 20%, and most of that gap was variance drag on the median
rather than tax mechanics (see the changelog). Every number regenerates from
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

- A deterministic research engine: same config + seed + environment = same result, serial or parallel (`--jobs N` splits paths across processes and returns identical numbers). Floating-point behavior varies across platforms and BLAS builds and can cross discrete trade thresholds, so official artifacts are generated only in pinned CI (runner image, CPython patch version, and numeric stack fixed in `.github/workflows/artifacts.yml` and `requirements-artifacts.txt`), and every manifest records the commit, worktree state, source-tree hash, platform, and full installed-package list that produced it.
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
pytest            # 71 tests: unit, regression, and property-based (hypothesis)
```

## Quick start

```python
from talsim import ScenarioConfig, run_sweep

cfg = ScenarioConfig()  # $1M, 10y, quarterly, 500 names, zero alpha, top 2026 federal rates
sweeps = run_sweep(cfg, ["100/0", "130/30"], n_paths=50)
for s in sweeps:
    print(
        s.book,
        f"median wealth ${s.median('ending_after_tax_wealth'):,.0f}",
        f"gross losses ${s.median('gross_losses_realized'):,.0f}",
        f"benefit used ${s.median('tax_benefit_used'):,.0f}",
    )
# 100/0  median wealth $1,636,689 gross losses $753,493 benefit used $126,447
# 130/30 median wealth $1,566,060 gross losses $2,415,247 benefit used $147,940
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
# wealth $847,766, TE 2.2%, turnover 3.3x, washed $0
```

Or from the command line:

```bash
talsim sweep --paths 200 --seed 7 --out results/
talsim scenarios --paths 100 --seed 7 --out results/
```

(`python -m talsim.cli` is equivalent to the `talsim` command. Add `--jobs N`
to run paths on N processes; results do not depend on it.)

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

## Model mechanics (v0.5.0)

- **Wash sales are enforced in the ledger**, both directions of the window, share-matched **in acquisition order with lot splitting**: when only part of a replacement lot matches, the matched shares become their own sublot carrying the transferred basis and a tacked TAX holding clock, while their actual acquisition date (which drives the wash window, the PIL 45-day test, and dividend qualification) is preserved separately. Short-side replacements have the deferred loss subtracted from their basis (sale proceeds), never added. **The window is an exact elapsed-day comparison**: at quarterly cadence a same-step repurchase washes and the next quarter, 91 days later, legally does not. Long-term character requires MORE than 365 days, per Pub 550. The policy layer independently avoids washes: it will not harvest a freshly bought name, it waits out the window before re-entering, redistributes blocked exposure to substitute names (capped at 2x each name's own target), and risk-driven reductions of recent buys sell gain lots first.
- **Exposure is constructed from post-trade state per side**, never signed drift, so short-to-long transitions land on target. A harvest floor prevents a side from flattening itself when every position is at a loss at once. Realized net exposure error is recorded per path.
- **The market generates ex-dividend price returns**; prices never drop on an ex-date. Dividends are paid in cash at the configured yield on long market value and **payments in lieu (PIL) are paid at the same yield on short market value**, so a long and a short in the same name net to zero and every net-100 book earns the same pre-tax net dividend income as 100/0 (`dividends_received`, `payments_in_lieu`, and `net_dividend_income` are reported per path). **Dividends are ordinary income**, split qualified/non-qualified by a day-based holding test (61 days, a proxy for the statutory 60-days-in-121 rule, correct at any cadence), taxed annually in their own buckets; capital losses never absorb them beyond the statutory ordinary offset.
- **PIL on a short closed within 45 days is capitalized into cover basis** (Pub 550). **PIL on a short open longer, plus margin debit interest, is investment interest expense** (IRC 163(d)): deducted at the ordinary rate against net investment income (interest, non-qualified dividends, and net short-term gain after netting, which includes the household's outside gains), with the excess carried forward. A loss harvester nets away its own short-term gains, so a household with small outside gains carries most of the deduction to the liquidation year. At quarterly cadence every short is open 91 days by its first possible close, so nothing capitalizes and everything is expensed. Borrow fees and the management fee are not deductible (miscellaneous itemized deductions, suspended since TCJA); qualified dividends and long-term gains count toward net investment income only under the 163(d)(4)(B) election, which is not modeled. `deduct_investment_interest=False` restores the pre-0.5 treatment.
- **Negative cash accrues debit interest** (default 6%); positive cash earns a configurable rate (default zero, deliberately conservative). Short proceeds fund the long extension, so there is no separate short-rebate line: `borrow_cost` is the net financing spread on the extension.
- **Margin** is a strategy-level maintenance test. The default `margin_model="portfolio"` is a risk-based requirement of `pm_stress` (15%, the regulatory stress for individual equities; brokers set higher house levels) times gross market value, the account type any book above 150/50 actually lives in: 250/150 needs 60% of equity and runs at full size. `margin_model="reg_t"` applies the FINRA Rule 4210 percentage floors instead (25% long / 30% short; the rule's per-share short minima for low-priced stocks and the 50% initial requirement are not modeled). Feasibility scaling **preserves net exposure**: an infeasible book keeps its long-only core and shrinks the long/short extension equally, so 250/150 at Reg T floors runs as roughly 233/133 (`extension_scale` reports the shrinkage) and every book in a sweep compares at the same market exposure. A deficiency during the path is cured by trading back to the compliant target fractions, with transaction costs and tax consequences; nonpositive equity ends the path in an explicit insolvent state. A "flag" mode records deficiencies without responding; its results should never be described as implementable. Actual average long and short exposures are reported per path.
- **Alpha**, when configured, enters as signal-proportional return drift calibrated at inception; the equal-weight 100/0 baseline has no active positions and receives none.
- **Universe and tracking error.** The default universe is 500 names (25% idiosyncratic vol, four sectors). The rank tilt puts every name in one tail or the other, so at 36 names a 150/50 held 13% of NAV in its largest position and carried active gross of 1.9 against equal weight, with a realized tracking error near 10% (20% for 250/150); at 500 names the largest position is under 2% of NAV and tracking error is about 1% for 100/0, 3% for 150/50, and 6% for 250/150. Tracking error is measured against an investable equal-weight portfolio of the same universe, and includes cost and tax drag. Most of the 36-name leverage penalty was variance drag on the median from that tracking error, not tax. The per-name no-trade band scales with the equal-weight slot (0.18 / n_assets of NAV, 0.5% at 36 names); a band fixed in NAV terms stops a large universe from trading at all. **Turnover** is one-sided (traded dollars / 2) over average NAV per year, excluding initial construction and terminal liquidation.

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

**0.5.0** — Fourth correctness release following a second external
review, from a practitioner who runs these strategies. Margin moves to a
portfolio-margin requirement by default (15% of gross market value), so
250/150 runs at full size instead of the Reg T-floor 233/133 (still
available as `margin_model="reg_t"`). Payments in lieu on shorts open more
than 45 days, plus debit interest, are now investment interest expense
deducted against net investment income with carryforward (IRC 163(d)); at
quarterly cadence the 45-day capitalization branch never fired, so 100% of
PIL previously got no tax treatment at all. Dividends received and net
dividend income are reported, documenting that longs and shorts earn and
pay the same yield and that PIL is not a pre-tax drag. The default universe
grows from 36 to 500 names: the 36-name rank tilt held up to 22% of NAV in
one name and produced tracking errors of 10 to 20%, and most of the
headline leverage penalty was variance drag from that construction rather
than tax mechanics. The per-name rebalance band now scales with the
equal-weight slot; the fixed 0.5%-of-NAV band exceeded every position at
500 names and silently left a 100/0 book in cash. `average_nav` is
reported per path. The legacy configuration (`n_assets=36,
rebalance_band=0.005, margin_model="reg_t",
deduct_investment_interest=False`) reproduces 0.4.1 results and is pinned
in the test suite. Results produced by 0.4.x should be discarded.

**0.4.1** — Performance release; results unchanged. `target_weights`
evaluates its scale grid in one vectorized pass (same grid and
tie-breaking, outputs bitwise identical to 0.4.0, verified against the
previous implementation in the test suite); `run_sweep` gains `n_jobs` and
the CLI gains `--jobs` for process-parallel paths with identical results;
official artifacts run with four jobs. A leveraged path is about three
times faster and sweeps scale with cores.

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
  version = {0.5.0},
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
