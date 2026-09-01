"""Regenerate the README results figure from the committed summary CSV.

Usage: python examples/make_readme_figure.py [results/leverage_sweep.csv]
"""

import sys
from pathlib import Path

import pandas as pd

from talsim.plotting import four_panel_from_summary

csv = Path(sys.argv[1] if len(sys.argv) > 1 else "results/leverage_sweep.csv")
four_panel_from_summary(pd.read_csv(csv), "docs/leverage_sweep.png")
print("wrote docs/leverage_sweep.png")
