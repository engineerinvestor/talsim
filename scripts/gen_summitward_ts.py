"""Generate web/src/lib/talsim-results.ts from talsim result CSVs."""

import json
from pathlib import Path

import pandas as pd

RESULTS = Path.home() / "Documents/code/talsim/results"
OUT = (
    Path.home()
    / "Documents/code/net-worth-tracker/web/src/lib/talsim-results.ts"
)

sweep = pd.read_csv(RESULTS / "leverage_sweep.csv")
manifest = json.loads((RESULTS / "leverage_sweep_manifest.json").read_text())

rows = []
for _, r in sweep.iterrows():
    rows.append(
        {
            "book": r["book"],
            "gross": round(float(r["gross_exposure"]), 2),
            "wealthP10": round(float(r["ending_after_tax_wealth_p10"])),
            "wealthMedian": round(float(r["ending_after_tax_wealth_median"])),
            "wealthP90": round(float(r["ending_after_tax_wealth_p90"])),
            "cagrMedian": round(float(r["after_tax_cagr_median"]), 4),
            "grossLosses": round(float(r["gross_losses_realized_median"])),
            "netLosses": round(float(r["net_loss_pre_liquidation_median"])),
            "benefitUsed": round(float(r["tax_benefit_used_median"])),
            "liquidationTax": round(float(r["liquidation_tax_median"])),
            "trackingError": round(float(r["tracking_error_median"]), 4),
            "maxDrawdown": round(float(r["max_drawdown_median"]), 4),
            "turnover": round(float(r["annual_turnover_median"]), 2),
            "fees": round(float(r["management_fees_median"])),
            "borrow": round(float(r["borrow_costs_median"])),
            "trading": round(float(r["transaction_costs_median"])),
            "paymentsInLieu": round(float(r["payments_in_lieu_median"])),
            "marginCallProb": round(float(r["margin_call_probability"]), 4),
        }
    )

scenario_rows = []
scenario_path = RESULTS / "scenario_comparison.csv"
if scenario_path.exists():
    scen = pd.read_csv(scenario_path)
    for _, r in scen.iterrows():
        scenario_rows.append(
            {
                "scenario": r["scenario"],
                "gross": round(float(r["gross_exposure"]), 2),
                "wealthMedian": round(float(r["ending_after_tax_wealth_median"])),
                "benefitUsed": round(float(r["tax_benefit_used_median"])),
                "marginCallProb": round(float(r["margin_call_probability"]), 4),
            }
        )

header = f"""// GENERATED from talsim result CSVs; do not hand-edit numbers.
// Source: talsim v{manifest["talsim_version"]} leverage sweep
// ({manifest["paths"]} common-random-number paths, base seed {manifest["base_seed"]},
// quarterly steps, 10 years, 36 synthetic assets, zero alpha, full liquidation).
// Regenerate: python -m talsim.cli sweep/scenarios, then scripts/gen_talsim_ts.py.

export interface TalsimBookSummary {{
  book: string;
  gross: number;
  wealthP10: number;
  wealthMedian: number;
  wealthP90: number;
  cagrMedian: number;
  grossLosses: number;
  netLosses: number;
  benefitUsed: number;
  liquidationTax: number;
  trackingError: number;
  maxDrawdown: number;
  turnover: number;
  fees: number;
  borrow: number;
  trading: number;
  paymentsInLieu: number;
  marginCallProb: number;
}}

export interface TalsimScenarioPoint {{
  scenario: string;
  gross: number;
  wealthMedian: number;
  benefitUsed: number;
  marginCallProb: number;
}}

export const TALSIM_RUN = {{
  version: "{manifest["talsim_version"]}",
  sweepPaths: {manifest["paths"]},
  scenarioPaths: 40,
  baseSeed: {manifest["base_seed"]},
  startingCapital: 1_000_000,
  years: 10,
}} as const;
"""


def ts_literal(obj: dict) -> str:
    parts = []
    for key, value in obj.items():
        if isinstance(value, str):
            parts.append(f'{key}: "{value}"')
        else:
            parts.append(f"{key}: {value}")
    return "  { " + ", ".join(parts) + " },"


lines = [header, "export const LEVERAGE_SWEEP: TalsimBookSummary[] = ["]
lines += [ts_literal(r) for r in rows]
lines.append("];\n")
lines.append("export const SCENARIO_COMPARISON: TalsimScenarioPoint[] = [")
lines += [ts_literal(r) for r in scenario_rows]
lines.append("];")

OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT} ({len(rows)} books, {len(scenario_rows)} scenario points)")
