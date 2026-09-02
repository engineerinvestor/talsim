"""Render the social-preview image for the Summitward TALS simulator guide.

Reads ``leverage_sweep.csv`` and draws median terminal after-tax wealth by
book with the 10th-90th percentile band, at the 1400x788 size the guide's
Open Graph metadata declares. Saved as WebP.

Usage:
    python scripts/gen_og_image.py [results_dir] [output_webp_path]
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from PIL import Image  # noqa: E402

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/results")
OUT = Path(
    sys.argv[2]
    if len(sys.argv) > 2
    else Path.home()
    / "Documents/code/net-worth-tracker/web/public/guides/tals-leverage-simulator-og.webp"
)

WIDTH, HEIGHT, DPI = 1400, 788, 100
INK = "#1f2937"
MUTED = "#6b7280"
LINE = "#1d4ed8"
BAND = "#93c5fd"
BG = "#fbfaf7"

sweep = pd.read_csv(RESULTS / "leverage_sweep.csv")
labels = [
    b if float(s) >= 0.999 else f"{b}*"
    for b, s in zip(sweep["book"], sweep["extension_scale"], strict=True)
]
x = range(len(sweep))
median = sweep["ending_after_tax_wealth_median"] / 1e6
p10 = sweep["ending_after_tax_wealth_p10"] / 1e6
p90 = sweep["ending_after_tax_wealth_p90"] / 1e6

fig, ax = plt.subplots(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.2)

ax.fill_between(x, p10, p90, color=BAND, alpha=0.45, linewidth=0, label="10th to 90th percentile")
ax.plot(x, median, color=LINE, linewidth=3.5, marker="o", markersize=10, label="Median")
ax.axhline(1.0, color=MUTED, linestyle=(0, (5, 4)), linewidth=1.5)
ax.text(2, 1.05, "starting capital $1M", color=MUTED, fontsize=13, ha="center")

for xi, m in zip(x, median, strict=True):
    ax.annotate(
        f"${m:.2f}M",
        (xi, m),
        textcoords="offset points",
        xytext=(0, 14),
        ha="center",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=16, color=INK)
ax.set_xlabel("Long / short exposure", fontsize=14, color=MUTED, labelpad=10)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.1f}M"))
ax.tick_params(axis="y", labelsize=13, colors=MUTED, length=0)
ax.tick_params(axis="x", length=0)
ax.set_ylim(0, float(p90.max()) * 1.1)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#d1d5db")
ax.grid(axis="y", color="#e5e7eb", linewidth=1)
ax.set_axisbelow(True)
ax.legend(loc="upper right", frameon=False, fontsize=13, labelcolor=INK)

fig.text(
    0.05,
    0.93,
    "Zero-alpha TALS simulation: more leverage, less after-tax wealth",
    fontsize=23,
    fontweight="bold",
    color=INK,
    ha="left",
    va="center",
)
fig.text(
    0.05,
    0.86,
    "Median terminal after-tax wealth, $1M start, 10 years, full liquidation, "
    "200 common market paths, talsim v0.4",
    fontsize=15,
    color=MUTED,
    ha="left",
    va="center",
)
fig.text(
    0.05,
    0.05,
    "*250/150 target; the FINRA-floor margin model runs it as roughly 233/133. "
    "Synthetic research results conditional on stated assumptions; not a forecast.",
    fontsize=12,
    color=MUTED,
    ha="left",
    va="center",
)

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=DPI, facecolor=BG)
buf.seek(0)
img = Image.open(buf).convert("RGB")
if img.size != (WIDTH, HEIGHT):
    img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT, "WEBP", quality=88, method=6)
print(f"wrote {OUT} {img.size}")
