"""talsim explorer: browse the CI-official results and run the engine live.

Deployed on Streamlit Community Cloud from the repository root. This app is
deliberately not part of the pytalsim wheel: it is a repo artifact, and its
"Official results" tab reads the committed, pinned-CI artifacts under
docs/results/ so it can never drift from the published numbers. Live runs
are small-sample by design (bounded paths behind a Run button) and are
labeled as such.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import talsim
from talsim import BOOK_PRESETS, ScenarioConfig, run_path
from talsim.cli import SCENARIOS, summarize
from talsim.simulation import SweepResult

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "docs" / "results"
BOOK_ORDER = list(BOOK_PRESETS)

st.set_page_config(page_title="talsim explorer", page_icon="📉", layout="wide")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


@st.cache_data
def load_manifest() -> dict:
    return json.loads((RESULTS / "leverage_sweep_manifest.json").read_text())


def usd(n: float) -> str:
    return f"${n:,.0f}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

manifest = load_manifest()
with st.sidebar:
    st.title("talsim explorer")
    st.caption(
        f"pytalsim v{talsim.__version__} · research simulator for "
        "tax-aware long-short (TALS) investing"
    )
    st.markdown(
        "- [Source on GitHub](https://github.com/engineerinvestor/talsim)\n"
        "- [PyPI: pytalsim](https://pypi.org/project/pytalsim/)\n"
        "- [API docs](https://engineerinvestor.github.io/talsim/)\n"
        "- [Full write-up with methodology]"
        "(https://summitward.com/learn/tals-leverage-simulator)"
    )
    st.divider()
    st.markdown(
        "**Official-run provenance**\n\n"
        f"- commit `{manifest['git_commit'][:9]}`\n"
        f"- {manifest['paths']} paths, seed {manifest['base_seed']}\n"
        f"- {manifest['platform'].split('-')[0]}, "
        f"Python {manifest['python_version']}, NumPy {manifest['numpy_version']}\n"
        f"- worktree dirty: {manifest['worktree_dirty']}"
    )
    st.divider()
    st.caption(
        "Synthetic research software, not tax, legal, or investment advice. "
        "Results are conditional on stated assumptions and are not evidence "
        "about any real strategy. The README documents every simplification."
    )

st.title("More harvested losses are not more wealth")
st.markdown(
    "A 250/150 tax-aware long-short book harvests several times the losses of "
    "plain long-only harvesting, and still ends behind it after netting, "
    "costs, risk, and the liquidation tax bill, in this zero-alpha synthetic "
    "experiment. Browse the pinned-CI official results, or run the actual "
    "engine on your own assumptions."
)

tab_official, tab_live = st.tabs(["Official results", "Run the engine"])


# ---------------------------------------------------------------------------
# Tab 1: official results
# ---------------------------------------------------------------------------

with tab_official:
    sweep = load_csv("leverage_sweep.csv")
    intervals = load_csv("leverage_sweep_intervals.csv")
    scenarios = load_csv("scenario_comparison.csv")

    st.subheader("Terminal after-tax wealth by leverage")
    st.caption(
        f"$1M for 10 years, quarterly steps, zero alpha, full liquidation; "
        f"medians with 10th-90th percentile bands across {manifest['paths']} "
        "common-random-number paths."
    )
    wealth = sweep[
        [
            "book",
            "ending_after_tax_wealth_p10",
            "ending_after_tax_wealth_median",
            "ending_after_tax_wealth_p90",
        ]
    ].rename(
        columns={
            "ending_after_tax_wealth_p10": "p10",
            "ending_after_tax_wealth_median": "median",
            "ending_after_tax_wealth_p90": "p90",
        }
    )
    band = (
        alt.Chart(wealth)
        .mark_area(opacity=0.15)
        .encode(
            x=alt.X("book:N", sort=BOOK_ORDER, title="Book"),
            y=alt.Y("p10:Q", title="Terminal after-tax wealth ($)"),
            y2="p90:Q",
        )
    )
    line = (
        alt.Chart(wealth)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("book:N", sort=BOOK_ORDER, title="Book"),
            y=alt.Y("median:Q"),
            tooltip=[
                alt.Tooltip("book:N"),
                alt.Tooltip("median:Q", format="$,.0f"),
                alt.Tooltip("p10:Q", format="$,.0f"),
                alt.Tooltip("p90:Q", format="$,.0f"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [1_000_000]}))
        .mark_rule(strokeDash=[4, 4], opacity=0.5)
        .encode(y="y:Q")
    )
    st.altair_chart((band + line + rule).properties(height=320), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Losses harvested vs. benefit used")
        losses = sweep.melt(
            id_vars="book",
            value_vars=["gross_losses_realized_median", "tax_benefit_used_median"],
            var_name="metric",
            value_name="dollars",
        )
        losses["metric"] = losses["metric"].map(
            {
                "gross_losses_realized_median": "Gross losses realized",
                "tax_benefit_used_median": "Tax benefit used",
            }
        )
        st.altair_chart(
            alt.Chart(losses)
            .mark_bar()
            .encode(
                x=alt.X("book:N", sort=BOOK_ORDER, title="Book"),
                xOffset="metric:N",
                y=alt.Y("dollars:Q", title="$ over horizon (median)"),
                color=alt.Color("metric:N", title=None),
                tooltip=[alt.Tooltip("metric:N"), alt.Tooltip("dollars:Q", format="$,.0f")],
            )
            .properties(height=300),
            width="stretch",
        )
    with right:
        st.subheader("Cumulative costs")
        cost_cols = {
            "management_fees_median": "Management fee",
            "borrow_costs_median": "Short borrow",
            "transaction_costs_median": "Trading",
            "payments_in_lieu_median": "Payments in lieu",
            "dividend_taxes_median": "Dividend taxes",
            "debit_interest_median": "Debit interest",
        }
        costs = sweep.melt(
            id_vars="book",
            value_vars=list(cost_cols),
            var_name="cost",
            value_name="dollars",
        )
        costs["cost"] = costs["cost"].map(cost_cols)
        st.altair_chart(
            alt.Chart(costs)
            .mark_bar()
            .encode(
                x=alt.X("book:N", sort=BOOK_ORDER, title="Book"),
                y=alt.Y("dollars:Q", title="$ over horizon (median)"),
                color=alt.Color("cost:N", title=None, sort=list(cost_cols.values())),
                tooltip=[alt.Tooltip("cost:N"), alt.Tooltip("dollars:Q", format="$,.0f")],
            )
            .properties(height=300),
            width="stretch",
        )

    st.subheader("Paired comparison vs. 100/0 on common paths")
    st.caption(
        "Bootstrap 95% intervals for the paired median difference and the "
        "probability of beating long-only; identical market paths per book."
    )
    paired = sweep[["book", "wealth_diff_vs_baseline_median", "prob_beats_baseline"]].merge(
        intervals[["book", "wealth_diff_ci_lo", "wealth_diff_ci_hi", "prob_ci_lo", "prob_ci_hi"]],
        on="book",
        how="left",
    )
    paired = paired[paired["book"] != "100/0"]
    display = pd.DataFrame(
        {
            "Book": paired["book"],
            "Median diff vs 100/0": paired["wealth_diff_vs_baseline_median"].map(usd),
            "95% CI": [
                f"{usd(lo)} to {usd(hi)}"
                for lo, hi in zip(
                    paired["wealth_diff_ci_lo"], paired["wealth_diff_ci_hi"], strict=True
                )
            ],
            "P(beats 100/0)": paired["prob_beats_baseline"].map(lambda p: f"{p:.0%}"),
            "P 95% CI": [
                f"{lo:.0%} to {hi:.0%}"
                for lo, hi in zip(paired["prob_ci_lo"], paired["prob_ci_hi"], strict=True)
            ],
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")

    st.subheader("Scenarios")
    scenario_names = list(scenarios["scenario"].unique())
    chosen = st.multiselect(
        "Compare fact patterns (100 paths each)", scenario_names, default=scenario_names
    )
    scen = scenarios[scenarios["scenario"].isin(chosen)][
        ["scenario", "book", "ending_after_tax_wealth_median"]
    ]
    st.altair_chart(
        alt.Chart(scen)
        .mark_line(point=True)
        .encode(
            x=alt.X("book:N", sort=BOOK_ORDER, title="Book"),
            y=alt.Y(
                "ending_after_tax_wealth_median:Q",
                title="Median terminal wealth ($)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("scenario:N", title=None, sort=scenario_names),
            tooltip=[
                alt.Tooltip("scenario:N"),
                alt.Tooltip("book:N"),
                alt.Tooltip("ending_after_tax_wealth_median:Q", format="$,.0f"),
            ],
        )
        .properties(height=340),
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Tab 2: live engine runs
# ---------------------------------------------------------------------------


def run_pair(
    book: str,
    paths: int,
    seed: int,
    overrides: dict,
) -> tuple[pd.DataFrame, float, float]:
    """Run `book` and the 100/0 baseline on common seeds. Returns
    (summary frame, paired median diff, prob of beating baseline)."""
    cfg = dataclasses.replace(ScenarioConfig(), **overrides)
    progress = st.progress(0.0, text="Simulating...")
    sweeps = []
    total = 2 * paths
    done = 0
    for book_name in ["100/0", book]:
        book_cfg = cfg.with_book(book_name)
        sweep_result = SweepResult(book=book_name, gross_exposure=book_cfg.gross_exposure)
        for p in range(paths):
            sweep_result.paths.append(run_path(book_cfg, seed=seed + p))
            done += 1
            progress.progress(done / total, text=f"Simulating... {done}/{total} paths")
        sweeps.append(sweep_result)
    progress.empty()
    frame = summarize(sweeps)
    diffs = [
        a.ending_after_tax_wealth - b.ending_after_tax_wealth
        for a, b in zip(sweeps[1].paths, sweeps[0].paths, strict=True)
    ]
    diffs.sort()
    median_diff = diffs[len(diffs) // 2]
    prob = sum(d > 0 for d in diffs) / len(diffs)
    return frame, median_diff, prob


with tab_live:
    st.caption(
        "Runs the actual pytalsim engine in this app, on a bounded number of "
        "paths. Small samples are noisy: treat these as directional, and "
        "compare against the 200-path official run in the first tab."
    )
    with st.form("live_run"):
        col1, col2, col3 = st.columns(3)
        with col1:
            book = st.selectbox("Book", [b for b in BOOK_ORDER if b != "100/0"], index=1)
            scenario_name = st.selectbox("Start from scenario", list(SCENARIOS))
            paths = st.slider("Paths (per book)", 10, 50, 25, step=5)
        with col2:
            gains = st.number_input(
                "Outside short-term gains per year ($)",
                min_value=0,
                max_value=2_000_000,
                value=100_000,
                step=25_000,
            )
            alpha_bps = st.slider("Assumed alpha (bps/yr at 150/50)", 0, 300, 0, step=25)
            years = st.slider("Horizon (years)", 3, 10, 10)
        with col3:
            fee_bps = st.slider("Management fee (bps/yr)", 0, 200, 45, step=5)
            borrow_bps = st.slider("Stock borrow (bps/yr)", 0, 400, 75, step=25)
            seed = st.number_input("Base seed", min_value=0, max_value=10_000, value=7)
        submitted = st.form_submit_button("Run", type="primary")

    if submitted:
        overrides = dict(SCENARIOS[scenario_name])
        overrides.update(
            {
                "outside_st_gains_annual": float(gains),
                "alpha_annual": alpha_bps / 10_000,
                "years": int(years),
                "management_fee": fee_bps / 10_000,
                "borrow_cost": borrow_bps / 10_000,
            }
        )
        # Year-keyed gain events from a scenario preset can fall outside a
        # shortened horizon; drop any that do.
        events = overrides.get("outside_st_gain_events") or {}
        overrides["outside_st_gain_events"] = {
            y: amt for y, amt in events.items() if y < int(years)
        }
        try:
            frame, median_diff, prob = run_pair(book, int(paths), int(seed), overrides)
        except ValueError as exc:
            st.error(f"Invalid configuration: {exc}")
        else:
            metric1, metric2, metric3 = st.columns(3)
            metric1.metric(f"{book} median wealth", usd(frame["ending_after_tax_wealth_median"][1]))
            metric2.metric("Paired median diff vs 100/0", usd(median_diff))
            metric3.metric("Beats 100/0 on", f"{prob:.0%} of paths")

            show_cols = {
                "book": "Book",
                "ending_after_tax_wealth_median": "Median wealth",
                "gross_losses_realized_median": "Gross losses",
                "tax_benefit_used_median": "Benefit used",
                "liquidation_tax_median": "Liquidation tax",
                "tracking_error_median": "Tracking error",
                "annual_turnover_median": "Turnover",
            }
            table = frame[list(show_cols)].rename(columns=show_cols).copy()
            for col in ["Median wealth", "Gross losses", "Benefit used", "Liquidation tax"]:
                table[col] = table[col].map(usd)
            table["Tracking error"] = table["Tracking error"].map(lambda x: f"{x:.1%}")
            table["Turnover"] = table["Turnover"].map(lambda x: f"{x:.1f}x")
            st.dataframe(table, hide_index=True, width="stretch")
            st.caption(
                f"{paths} common-random-number paths per book, base seed {seed}, "
                "full liquidation at the horizon, federal top rates, no state "
                "tax. Alpha scales with active gross exposure and is zero for "
                "the 100/0 baseline by construction."
            )
