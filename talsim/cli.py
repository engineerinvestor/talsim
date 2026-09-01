"""Reproducible command-line runs.

Examples:

    python -m talsim.cli sweep --paths 80 --seed 7 --out results/
    python -m talsim.cli scenarios --paths 40 --seed 7 --out results/

Every run writes a manifest (config, seed, package version) next to its
results so a CSV can always be traced back to the assumptions that made it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import pandas as pd

from . import __version__
from .config import BOOK_PRESETS, ScenarioConfig
from .simulation import run_sweep

SUMMARY_ATTRS = [
    "ending_after_tax_wealth",
    "after_tax_cagr",
    "gross_losses_realized",
    "net_loss_pre_liquidation",
    "tax_benefit_used",
    "unused_loss_carry",
    "liquidation_tax",
    "tracking_error",
    "max_drawdown",
    "annual_turnover",
    "management_fees",
    "borrow_costs",
    "transaction_costs",
    "payments_in_lieu",
    "min_margin_excess_ratio",
]


def summarize(sweeps) -> pd.DataFrame:
    rows = []
    for sweep in sweeps:
        row: dict[str, float | str] = {
            "book": sweep.book,
            "gross_exposure": sweep.gross_exposure,
            "margin_call_probability": sweep.margin_call_probability(),
        }
        for attr in SUMMARY_ATTRS:
            row[f"{attr}_p10"] = sweep.percentile(attr, 10)
            row[f"{attr}_median"] = sweep.median(attr)
            row[f"{attr}_p90"] = sweep.percentile(attr, 90)
        rows.append(row)
    return pd.DataFrame(rows)


def write_manifest(out: Path, cfg: ScenarioConfig, args: argparse.Namespace, name: str) -> None:
    manifest = {
        "talsim_version": __version__,
        "command": name,
        "paths": args.paths,
        "base_seed": args.seed,
        "config": dataclasses.asdict(cfg),
    }
    (out / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def cmd_sweep(args: argparse.Namespace) -> None:
    cfg = ScenarioConfig()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sweeps = run_sweep(cfg, list(BOOK_PRESETS), args.paths, base_seed=args.seed)
    df = summarize(sweeps)
    df.to_csv(out / "leverage_sweep.csv", index=False)
    write_manifest(out, cfg, args, "leverage_sweep")
    print(df[["book", "ending_after_tax_wealth_median", "margin_call_probability"]])


SCENARIOS: dict[str, dict] = {
    "Annual $100k ST gains, zero alpha": {},
    "No outside gains, zero alpha": {"outside_st_gains_annual": 0.0},
    "$500k ST gain in year 3, zero alpha": {
        "outside_st_gains_annual": 0.0,
        "outside_st_gain_events": {2: 500_000.0},
    },
    "Annual $100k ST gains, +75 bps alpha": {"alpha_annual": 0.0075},
    "Annual gains, zero alpha, high costs": {
        "management_fee": 0.0095,
        "borrow_cost": 0.0175,
        "transaction_cost": 0.0015,
    },
}


def cmd_scenarios(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, overrides in SCENARIOS.items():
        cfg = dataclasses.replace(ScenarioConfig(), **overrides)
        sweeps = run_sweep(cfg, list(BOOK_PRESETS), args.paths, base_seed=args.seed)
        df = summarize(sweeps)
        df.insert(0, "scenario", name)
        frames.append(df)
        print(f"done: {name}")
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(out / "scenario_comparison.csv", index=False)
    write_manifest(out, ScenarioConfig(), args, "scenario_comparison")


def main() -> None:
    parser = argparse.ArgumentParser(prog="talsim")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in [("sweep", cmd_sweep), ("scenarios", cmd_scenarios)]:
        p = sub.add_parser(name)
        p.add_argument("--paths", type=int, default=80 if name == "sweep" else 40)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--out", type=str, default="results")
        p.set_defaults(fn=fn)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
