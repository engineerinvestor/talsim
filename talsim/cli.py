"""Reproducible command-line runs.

Examples:

    python -m talsim.cli sweep --paths 200 --seed 7 --out results/
    python -m talsim.cli scenarios --paths 100 --seed 7 --out results/

Every run writes, next to its summary CSV:

- a path-level CSV (one row per book per path, with the seed), so anyone
  can recompute percentiles, paired differences, or their own statistics;
- a manifest recording the package version, git commit, Python and NumPy
  versions, full config including per-scenario overrides, and a SHA-256
  checksum of each CSV written.

The sweep summary also reports paired differences versus the 100/0 book on
common random numbers: the median wealth difference and the fraction of
paths on which each book beat 100/0.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import BOOK_PRESETS, ScenarioConfig
from .simulation import run_sweep

SUMMARY_ATTRS = [
    "ending_after_tax_wealth",
    "after_tax_cagr",
    "gross_losses_realized",
    "gross_losses_liquidation",
    "disallowed_wash_losses",
    "net_realized_pre_liquidation",
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
    "dividend_taxes",
    "debit_interest",
    "min_margin_excess_ratio",
    "avg_long_exposure",
    "avg_short_exposure",
    "max_net_exposure_error",
]

PATH_ATTRS = SUMMARY_ATTRS + [
    "maintenance_deficiency_observed",
    "feasible_at_inception",
    "insolvent",
    "deleverage_events",
    "extension_scale",
    "yearly_taxes_paid",
]


def summarize(sweeps) -> pd.DataFrame:
    baseline = sweeps[0]
    rows = []
    for sweep in sweeps:
        row: dict[str, float | str | bool] = {
            "book": sweep.book,
            "gross_exposure": sweep.gross_exposure,
            "maintenance_deficiency_probability": sweep.deficiency_probability(),
            "feasible_at_inception": sweep.paths[0].feasible_at_inception,
            "deleverage_events_median": float(
                np.median([p.deleverage_events for p in sweep.paths])
            ),
            "extension_scale": sweep.paths[0].extension_scale,
            "insolvency_probability": float(np.mean([p.insolvent for p in sweep.paths])),
        }
        for attr in SUMMARY_ATTRS:
            row[f"{attr}_p10"] = sweep.percentile(attr, 10)
            row[f"{attr}_median"] = sweep.median(attr)
            row[f"{attr}_p90"] = sweep.percentile(attr, 90)
        # Paired comparison vs the first (100/0) book on common paths.
        diffs = [
            a.ending_after_tax_wealth - b.ending_after_tax_wealth
            for a, b in zip(sweep.paths, baseline.paths, strict=True)
        ]
        row["wealth_diff_vs_baseline_median"] = float(np.median(diffs))
        row["wealth_diff_vs_baseline_p10"] = float(np.percentile(diffs, 10))
        row["wealth_diff_vs_baseline_p90"] = float(np.percentile(diffs, 90))
        row["prob_beats_baseline"] = float(np.mean([d > 0 for d in diffs]))
        rows.append(row)
    return pd.DataFrame(rows)


def path_frame(sweeps, base_seed: int) -> pd.DataFrame:
    rows = []
    for sweep in sweeps:
        for i, p in enumerate(sweep.paths):
            row: dict[str, float | str | bool | int] = {
                "book": sweep.book,
                "gross_exposure": sweep.gross_exposure,
                "path": i,
                "seed": base_seed + i,
            }
            for attr in PATH_ATTRS:
                row[attr] = getattr(p, attr)
            rows.append(row)
    return pd.DataFrame(rows)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(
    out: Path,
    name: str,
    args: argparse.Namespace,
    configs: dict[str, ScenarioConfig],
    files: list[Path],
) -> None:
    manifest = {
        "talsim_version": __version__,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "command": name,
        "paths": args.paths,
        "base_seed": args.seed,
        "configs": {label: dataclasses.asdict(cfg) for label, cfg in configs.items()},
        "file_checksums": {f.name: _sha256(f) for f in files if f.exists()},
    }
    (out / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def cmd_sweep(args: argparse.Namespace) -> None:
    cfg = ScenarioConfig()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sweeps = run_sweep(cfg, list(BOOK_PRESETS), args.paths, base_seed=args.seed)
    summary_path = out / "leverage_sweep.csv"
    paths_path = out / "leverage_sweep_paths.csv"
    summarize(sweeps).to_csv(summary_path, index=False)
    path_frame(sweeps, args.seed).to_csv(paths_path, index=False)
    write_manifest(out, "leverage_sweep", args, {"default": cfg}, [summary_path, paths_path])
    df = pd.read_csv(summary_path)
    print(
        df[
            [
                "book",
                "ending_after_tax_wealth_median",
                "wealth_diff_vs_baseline_median",
                "prob_beats_baseline",
                "maintenance_deficiency_probability",
            ]
        ]
    )


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
    path_frames = []
    configs: dict[str, ScenarioConfig] = {}
    for name, overrides in SCENARIOS.items():
        cfg = dataclasses.replace(ScenarioConfig(), **overrides)
        configs[name] = cfg
        sweeps = run_sweep(cfg, list(BOOK_PRESETS), args.paths, base_seed=args.seed)
        df = summarize(sweeps)
        df.insert(0, "scenario", name)
        frames.append(df)
        pf = path_frame(sweeps, args.seed)
        pf.insert(0, "scenario", name)
        path_frames.append(pf)
        print(f"done: {name}")
    summary_path = out / "scenario_comparison.csv"
    paths_path = out / "scenario_comparison_paths.csv"
    pd.concat(frames, ignore_index=True).to_csv(summary_path, index=False)
    pd.concat(path_frames, ignore_index=True).to_csv(paths_path, index=False)
    write_manifest(out, "scenario_comparison", args, configs, [summary_path, paths_path])


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(prog="talsim")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in [("sweep", cmd_sweep), ("scenarios", cmd_scenarios)]:
        p = sub.add_parser(name)
        p.add_argument("--paths", type=positive_int, default=200 if name == "sweep" else 100)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--out", type=str, default="results")
        p.set_defaults(fn=fn)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
