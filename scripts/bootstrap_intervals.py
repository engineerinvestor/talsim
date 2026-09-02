"""Sampling intervals for the paired leverage comparisons.

Reads the path-level CSVs written by ``talsim sweep`` and ``talsim scenarios``,
pairs every book with the 100/0 baseline on the same path index (common random
numbers), and writes one row per (scenario, book) with:

- the median paired wealth difference and a 95% percentile-bootstrap interval
  for that median (2000 resamples, ``numpy.random.default_rng(0)``);
- the number of paths on which the book beat the baseline, the resulting win
  probability, and its 95% Wilson interval.

These intervals measure Monte Carlo sampling noise within the model at the
stated path count. They say nothing about whether the model resembles real
markets, real trading, or a real tax return.

Usage:
    python scripts/bootstrap_intervals.py [results_dir]
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/results")
BASELINE = "100/0"
RESAMPLES = 2000
SEED = 0
Z = 1.959964

FIELDS = [
    "scenario",
    "book",
    "n_paths",
    "wealth_diff_median",
    "wealth_diff_ci_lo",
    "wealth_diff_ci_hi",
    "wins",
    "prob_beats",
    "prob_ci_lo",
    "prob_ci_hi",
]


def wilson(wins: int, n: int) -> tuple[float, float]:
    p = wins / n
    denom = 1 + Z * Z / n
    centre = p + Z * Z / (2 * n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (centre - half) / denom, (centre + half) / denom


def bootstrap_median(diffs: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(diffs), size=(RESAMPLES, len(diffs)))
    medians = np.median(diffs[idx], axis=1)
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return float(lo), float(hi)


def paired_rows(path_csv: Path, group_key: str | None) -> list[dict]:
    wealth: dict[tuple[str, str], dict[int, float]] = {}
    with path_csv.open() as f:
        for r in csv.DictReader(f):
            key = (r[group_key] if group_key else "", r["book"])
            wealth.setdefault(key, {})[int(r["path"])] = float(r["ending_after_tax_wealth"])

    rng = np.random.default_rng(SEED)
    out = []
    for (scenario, book), by_path in wealth.items():
        if book == BASELINE:
            continue
        base = wealth[(scenario, BASELINE)]
        paths = sorted(by_path)
        diffs = np.array([by_path[p] - base[p] for p in paths])
        lo, hi = bootstrap_median(diffs, rng)
        wins = int(np.sum(diffs > 0))
        n = len(diffs)
        p_lo, p_hi = wilson(wins, n)
        out.append(
            {
                "scenario": scenario,
                "book": book,
                "n_paths": n,
                "wealth_diff_median": round(float(np.median(diffs)), 2),
                "wealth_diff_ci_lo": round(lo, 2),
                "wealth_diff_ci_hi": round(hi, 2),
                "wins": wins,
                "prob_beats": round(wins / n, 4),
                "prob_ci_lo": round(p_lo, 4),
                "prob_ci_hi": round(p_hi, 4),
            }
        )
    return out


def write(rows: list[dict], out_csv: Path) -> None:
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} rows)")
    for r in rows:
        label = f"{r['scenario'][:36]:36s} {r['book']:8s}" if r["scenario"] else f"{r['book']:8s}"
        print(
            f"  {label} median {r['wealth_diff_median']:>12,.0f} "
            f"[{r['wealth_diff_ci_lo']:,.0f}, {r['wealth_diff_ci_hi']:,.0f}]  "
            f"beats {r['prob_beats']:.0%} [{r['prob_ci_lo']:.0%}, {r['prob_ci_hi']:.0%}]"
        )


if __name__ == "__main__":
    write(
        paired_rows(RESULTS / "leverage_sweep_paths.csv", None),
        RESULTS / "leverage_sweep_intervals.csv",
    )
    write(
        paired_rows(RESULTS / "scenario_comparison_paths.csv", "scenario"),
        RESULTS / "scenario_comparison_intervals.csv",
    )
