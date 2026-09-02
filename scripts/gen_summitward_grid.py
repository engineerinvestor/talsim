"""Generate Summitward's talsim-grid.ts module from run_grid.py results.

Usage:
    python scripts/gen_summitward_grid.py [results_dir] [output_ts_path]

Validates the grid manifest against the CSV checksums first. Dollar columns
are divided by the reference capital so the browser can rescale them to the
reader's own starting capital; every value is rounded to four significant
digits to keep the shipped module small.

The values array is row-major in axis order (gains ratio, cost tier, alpha,
horizon, bracket, book), then field order. ``talsim-grid-lookup.ts`` in the
Summitward repo indexes it; keep the two in sync.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
OUT = Path(
    sys.argv[2]
    if len(sys.argv) > 2
    else Path.home() / "Documents/code/net-worth-tracker/web/src/lib/talsim-grid.ts"
)

# (TS field name, CSV column, scale by reference capital?)
FIELDS: list[tuple[str, str, bool]] = [
    ("wealthP10", "ending_after_tax_wealth_p10", True),
    ("wealthMedian", "ending_after_tax_wealth_median", True),
    ("wealthP90", "ending_after_tax_wealth_p90", True),
    ("wealthDiffP10", "wealth_diff_vs_baseline_p10", True),
    ("wealthDiffMedian", "wealth_diff_vs_baseline_median", True),
    ("wealthDiffP90", "wealth_diff_vs_baseline_p90", True),
    ("probBeatsBaseline", "prob_beats_baseline", False),
    ("benefitUsed", "tax_benefit_used_median", True),
    ("grossLosses", "gross_losses_realized_median", True),
    ("totalCosts", "total_costs_median", True),
    ("liquidationTax", "liquidation_tax_median", True),
    ("maxDrawdown", "max_drawdown_median", False),
    ("trackingError", "tracking_error_median", False),
    ("deficiencyProb", "maintenance_deficiency_probability", False),
    ("extensionScale", "extension_scale", False),
]


def load_validated(name: str) -> dict:
    manifest = json.loads((RESULTS / f"{name}_manifest.json").read_text())
    for fname, recorded in manifest["file_checksums"].items():
        actual = hashlib.sha256((RESULTS / fname).read_bytes()).hexdigest()
        if actual != recorded:
            raise SystemExit(f"checksum mismatch for {fname}: manifest does not match file")
    return manifest


def sig(value: float, digits: int = 4) -> str:
    if value == 0:
        return "0"
    text = f"{value:.{digits}g}"
    if "e" in text:
        text = f"{float(text):.10f}".rstrip("0").rstrip(".")
    return text


manifest = load_validated("grid")
axes = manifest["grid_axes"]
capital = float(axes["reference_capital"])
df = pd.read_csv(RESULTS / "grid_summary.csv")

books: list[str] = axes["books"]
gains: list[float] = axes["gains_ratios"]
costs: list[str] = axes["cost_tier_order"]
alphas: list[float] = axes["alphas"]
horizons: list[int] = axes["horizons"]
brackets: list[str] = axes["bracket_order"]

indexed = df.set_index(["gains_ratio", "cost_tier", "alpha", "horizon", "bracket", "book"])
expected = len(gains) * len(costs) * len(alphas) * len(horizons) * len(brackets) * len(books)
if len(indexed) != expected:
    raise SystemExit(f"grid has {len(indexed)} rows, expected {expected}")

values: list[str] = []
for g in gains:
    for c in costs:
        for a in alphas:
            for h in horizons:
                for b in brackets:
                    for book in books:
                        row = indexed.loc[(g, c, a, h, b, book)]
                        for _, col, scaled in FIELDS:
                            v = float(row[col])
                            values.append(sig(v / capital if scaled else v))

field_names = [name for name, _, _ in FIELDS]


def camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part.capitalize() for part in rest)


def ts_record(order: list[str], table: dict[str, dict[str, float]]) -> str:
    lines = ["{"]
    for key in order:
        inner = ", ".join(f"{camel(k)}: {v:g}" for k, v in table[key].items())
        lines.append(f"  {key}: {{ {inner} }},")
    lines.append("}")
    return "\n".join(lines)


cost_ts = ts_record(costs, axes["cost_tiers"])
bracket_ts = ts_record(brackets, axes["brackets"])


header = f"""// GENERATED from talsim grid CSVs; do not hand-edit numbers.
// Source: talsim v{manifest["talsim_version"]} (git {manifest["git_commit"][:9]}),
// {manifest["paths"]} common-random-number paths per cell, base seed
// {manifest["base_seed"]}, {len(indexed)} book-cells, ${capital / 1e6:.0f}M reference capital.
// Quarterly steps, 36 synthetic assets, full liquidation at horizon end.
// Regenerate: python scripts/run_grid.py, then
// scripts/gen_summitward_grid.py in the talsim repo.

export const TALSIM_GRID_RUN = {{
  version: "{manifest["talsim_version"]}",
  gitCommit: "{manifest["git_commit"][:9]}",
  paths: {manifest["paths"]},
  baseSeed: {manifest["base_seed"]},
  referenceCapital: {int(capital):_},
}} as const;

/** Axis values in storage order. Values are row-major in this order, then field order. */
export const TALSIM_GRID_AXES = {{
  books: {json.dumps(books)},
  gainsRatios: {json.dumps(gains)},
  costTiers: {json.dumps(costs)},
  alphas: {json.dumps(alphas)},
  horizons: {json.dumps(horizons)},
  brackets: {json.dumps(brackets)},
}} as const;

export type TalsimGridBook = (typeof TALSIM_GRID_AXES.books)[number];
export type TalsimGridCostTier = (typeof TALSIM_GRID_AXES.costTiers)[number];
export type TalsimGridBracket = (typeof TALSIM_GRID_AXES.brackets)[number];

/** Annualized cost assumptions behind each tier (decimal rates). */
export const TALSIM_GRID_COST_TIERS: Record<
  TalsimGridCostTier,
  {{ managementFee: number; borrowCost: number; transactionCost: number; debitRate: number }}
> = {cost_ts};

/** Federal rate assumptions behind each bracket (decimal rates). */
export const TALSIM_GRID_BRACKETS: Record<
  TalsimGridBracket,
  {{ stRate: number; ltRate: number; ordinaryRate: number }}
> = {bracket_ts};

/** Dollar fields are ratios of reference capital; rates and probabilities are decimals. */
export const TALSIM_GRID_FIELDS = {json.dumps(field_names)} as const;

export type TalsimGridField = (typeof TALSIM_GRID_FIELDS)[number];
"""

body_lines = []
per_line = 15
for i in range(0, len(values), per_line):
    body_lines.append("  " + ", ".join(values[i : i + per_line]) + ",")
body = "export const TALSIM_GRID_VALUES: number[] = [\n" + "\n".join(body_lines) + "\n];\n"

OUT.write_text(header + "\n" + body)
size_kb = OUT.stat().st_size / 1024
print(f"wrote {OUT} ({len(indexed)} book-cells, {len(values)} values, {size_kb:.0f} KB)")
