"""talsim: research simulator for tax-aware long-short portfolio strategies.

Synthetic, educational, and explicitly not tax advice. See README for the
list of simplifications and the boundary between this research model and
tax-return-grade accounting.
"""

from .config import BOOK_PRESETS, ScenarioConfig
from .lots import Ledger
from .simulation import PathResult, SweepResult, run_path, run_sweep

__version__ = "0.2.0"

__all__ = [
    "BOOK_PRESETS",
    "Ledger",
    "PathResult",
    "ScenarioConfig",
    "SweepResult",
    "run_path",
    "run_sweep",
    "__version__",
]
