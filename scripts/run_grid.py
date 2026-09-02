"""Run the precomputed leverage grid that powers Summitward's in-page calculator.

Usage:
    python scripts/run_grid.py --paths 100 --seed 7 --workers 8 --out results/

The grid spans every input that changes the leverage answer and cannot be
scaled analytically: book, outside short-term gains as a ratio of starting
capital, cost tier, alpha, horizon, and tax bracket. Starting capital is held
at a $1M reference because every rate and cost in ``ScenarioConfig`` is a
ratio except the $3,000 ordinary-income offset; the browser rescales dollar
outputs to the reader's capital.

Every cell runs all five books on common random numbers (path p uses seed
base+p in every cell and book), so cross-book and cross-cell differences sit
on identical market paths.

Outputs, next to each other in ``--out``:

- ``grid_summary.csv``: one row per (cell, book) with the same columns as
  ``leverage_sweep.csv`` plus the cell keys and a ``total_costs`` triple;
- ``grid_paths.csv.gz``: one row per (cell, book, path) with the seed;
- ``grid_manifest.json``: the usual provenance record plus the axis values,
  worker count, and wall time.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from talsim.cli import path_frame, positive_int, summarize, write_manifest
from talsim.config import BOOK_PRESETS, ScenarioConfig
from talsim.simulation import SweepResult, run_path

REFERENCE_CAPITAL = 1_000_000.0

BOOKS: list[str] = list(BOOK_PRESETS)
GAINS_RATIOS: list[float] = [0.0, 0.025, 0.05, 0.10, 0.20]
COST_TIERS: dict[str, dict[str, float]] = {
    "low": {
        "management_fee": 0.0025,
        "borrow_cost": 0.0040,
        "transaction_cost": 0.0003,
        "debit_rate": 0.05,
    },
    "base": {
        "management_fee": 0.0045,
        "borrow_cost": 0.0075,
        "transaction_cost": 0.0005,
        "debit_rate": 0.06,
    },
    "high": {
        "management_fee": 0.0095,
        "borrow_cost": 0.0175,
        "transaction_cost": 0.0015,
        "debit_rate": 0.07,
    },
}
ALPHAS: list[float] = [0.0, 0.0075]
HORIZONS: list[int] = [5, 10, 20]
BRACKETS: dict[str, dict[str, float]] = {
    "top": {"st_rate": 0.408, "lt_rate": 0.238, "ordinary_rate": 0.408},
    "mid": {"st_rate": 0.24, "lt_rate": 0.15, "ordinary_rate": 0.24},
}

CELL_KEYS = ["gains_ratio", "cost_tier", "alpha", "horizon", "bracket"]
COST_ATTRS = [
    "management_fees",
    "borrow_costs",
    "transaction_costs",
    "payments_in_lieu",
    "debit_interest",
]


def cell_label(gains: float, cost: str, alpha: float, horizon: int, bracket: str) -> str:
    return f"g{gains:g}_{cost}_a{alpha:g}_y{horizon}_{bracket}"


def cell_config(
    gains: float, cost: str, alpha: float, horizon: int, bracket: str
) -> ScenarioConfig:
    return dataclasses.replace(
        ScenarioConfig(),
        starting_capital=REFERENCE_CAPITAL,
        outside_st_gains_annual=gains * REFERENCE_CAPITAL,
        alpha_annual=alpha,
        years=horizon,
        **COST_TIERS[cost],
        **BRACKETS[bracket],
    )


def all_cells() -> list[tuple[float, str, float, int, str]]:
    return list(itertools.product(GAINS_RATIOS, COST_TIERS, ALPHAS, HORIZONS, BRACKETS))


def _run_book(task: tuple[str, ScenarioConfig, str, int, int]) -> tuple[str, SweepResult]:
    label, cfg, book, n_paths, base_seed = task
    book_cfg = cfg.with_book(book)
    sweep = SweepResult(book=book, gross_exposure=book_cfg.gross_exposure)
    for p in range(n_paths):
        sweep.paths.append(run_path(book_cfg, seed=base_seed + p))
    return label, sweep


def _total_costs(sweeps: list[SweepResult]) -> pd.DataFrame:
    rows = []
    for sweep in sweeps:
        totals = [sum(getattr(p, a) for a in COST_ATTRS) for p in sweep.paths]
        rows.append(
            {
                "total_costs_p10": float(np.percentile(totals, 10)),
                "total_costs_median": float(np.median(totals)),
                "total_costs_p90": float(np.percentile(totals, 90)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(prog="run_grid")
    parser.add_argument("--paths", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=positive_int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cells = all_cells()
    configs = {cell_label(*c): cell_config(*c) for c in cells}
    tasks = [
        (label, cfg, book, args.paths, args.seed)
        for label, cfg in configs.items()
        for book in BOOKS
    ]
    print(f"{len(cells)} cells x {len(BOOKS)} books x {args.paths} paths on {args.workers} workers")

    started = time.perf_counter()
    by_cell: dict[str, dict[str, SweepResult]] = {label: {} for label in configs}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for label, sweep in pool.map(_run_book, tasks, chunksize=1):
            by_cell[label][sweep.book] = sweep
            done += 1
            if done % 25 == 0 or done == len(tasks):
                elapsed = time.perf_counter() - started
                print(f"  {done}/{len(tasks)} book-cells, {elapsed / 60:.1f} min elapsed")
    wall = time.perf_counter() - started

    frames = []
    path_frames = []
    for cell in cells:
        label = cell_label(*cell)
        sweeps = [by_cell[label][book] for book in BOOKS]
        df = pd.concat([summarize(sweeps), _total_costs(sweeps)], axis=1)
        pf = path_frame(sweeps, args.seed)
        for key, value in reversed(list(zip(CELL_KEYS, cell, strict=True))):
            df.insert(0, key, value)
            pf.insert(0, key, value)
        frames.append(df)
        path_frames.append(pf)

    summary_path = out / "grid_summary.csv"
    paths_path = out / "grid_paths.csv.gz"
    pd.concat(frames, ignore_index=True).to_csv(summary_path, index=False)
    pd.concat(path_frames, ignore_index=True).to_csv(paths_path, index=False, compression="gzip")
    write_manifest(out, "grid", args, configs, [summary_path, paths_path])

    manifest_path = out / "grid_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["grid_axes"] = {
        "reference_capital": REFERENCE_CAPITAL,
        "books": BOOKS,
        "gains_ratios": GAINS_RATIOS,
        "cost_tiers": COST_TIERS,
        "cost_tier_order": list(COST_TIERS),
        "alphas": ALPHAS,
        "horizons": HORIZONS,
        "brackets": BRACKETS,
        "bracket_order": list(BRACKETS),
    }
    manifest["workers"] = args.workers
    manifest["wall_seconds"] = round(wall, 1)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {summary_path} ({len(cells)} cells) in {wall / 60:.1f} min")


if __name__ == "__main__":
    main()
