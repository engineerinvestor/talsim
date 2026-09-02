"""Scenario configuration for talsim simulations.

Every assumption that drives a result lives here, so a saved config plus a
seed reproduces a run exactly. Rates are decimals (0.408 = 40.8%), horizons
are years, and exposures are fractions of net asset value (1.3 long / 0.3
short is a 130/30 book).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Preset long/short books, keyed by conventional name. Net exposure is 1.0
# (fully invested, beta-one) in every preset; only gross varies.
BOOK_PRESETS: dict[str, tuple[float, float]] = {
    "100/0": (1.0, 0.0),
    "130/30": (1.3, 0.3),
    "150/50": (1.5, 0.5),
    "200/100": (2.0, 1.0),
    "250/150": (2.5, 1.5),
}


@dataclass
class ScenarioConfig:
    # Capital and horizon
    starting_capital: float = 1_000_000.0
    years: int = 10
    steps_per_year: int = 4  # quarterly rebalance and harvest

    # Universe and synthetic market (annualized parameters)
    n_assets: int = 36
    n_sectors: int = 4
    market_drift: float = 0.06
    market_vol: float = 0.16
    sector_vol: float = 0.08
    idio_vol: float = 0.25
    dividend_yield: float = 0.013
    signal_autocorr: float = 0.90  # AR(1) persistence of the cross-sectional signal

    # Book
    long_exposure: float = 1.0
    short_exposure: float = 0.0

    # Alpha assumption. Applied as deterministic drift on the active book,
    # scaled linearly with active gross exposure relative to the reference
    # (150/50 has active gross 1.0). Zero by default on purpose: with any
    # positive alpha, a leverage comparison silently becomes an alpha study.
    alpha_annual: float = 0.0
    alpha_reference_active_gross: float = 1.0

    # Tax rates (top 2026 federal rates including 3.8% NIIT)
    st_rate: float = 0.408
    lt_rate: float = 0.238
    ordinary_rate: float = 0.408
    ordinary_offset_limit: float = 3_000.0

    # Outside gain schedule: recurring annual short-term gains, plus optional
    # one-time events keyed by year index (0-based).
    outside_st_gains_annual: float = 100_000.0
    outside_st_gain_events: dict[int, float] = field(default_factory=dict)

    # Costs (annualized where applicable)
    management_fee: float = 0.0045  # on net asset value
    borrow_cost: float = 0.0075  # on short market value
    transaction_cost: float = 0.0005  # per dollar traded

    # Harvesting policy
    harvest_threshold: float = 0.02  # realize a lot's loss when >2% below basis
    rebalance_band: float = 0.005  # ignore target drift smaller than 0.5% NAV
    wash_window_days: int = 30  # statutory window, compared in exact elapsed days
    # A side never harvests itself below this fraction of its exposure
    # target: when every short is at a loss at once, realizing them all
    # would flatten the book for a wash window. Smallest losses defer first.
    harvest_exposure_floor: float = 0.7

    # Financing: negative cash accrues debit interest; positive cash earns
    # cash_rate (default zero, a deliberate conservatism).
    debit_rate: float = 0.06
    cash_rate: float = 0.0

    # Margin model: strategy-level maintenance test at FINRA Rule 4210 floor
    # levels (brokers set higher house requirements). When breached, the
    # engine force-deleverages proportionally (with taxes and trading costs)
    # instead of flagging and continuing on impossible capital.
    long_maintenance: float = 0.25
    short_maintenance: float = 0.30
    margin_response: str = "deleverage"  # "deleverage" | "flag"
    deleverage_buffer: float = 0.98  # restore equity to req/buffer coverage

    def __post_init__(self) -> None:
        if self.starting_capital <= 0:
            raise ValueError("starting_capital must be positive")
        if self.years < 1 or self.steps_per_year < 1:
            raise ValueError("years and steps_per_year must be >= 1")
        if self.long_exposure < 0 or self.short_exposure < 0:
            raise ValueError("exposures must be non-negative")
        if self.long_exposure - self.short_exposure <= 0:
            raise ValueError("net exposure must be positive")
        if self.n_assets < 4:
            raise ValueError("n_assets must be >= 4")
        for rate in (self.st_rate, self.lt_rate, self.ordinary_rate):
            if not 0 <= rate < 1:
                raise ValueError("tax rates must be in [0, 1)")
        if self.margin_response not in ("deleverage", "flag"):
            raise ValueError("margin_response must be 'deleverage' or 'flag'")
        if self.wash_window_days < 1:
            raise ValueError("wash_window_days must be >= 1")
        if self.n_sectors < 1:
            raise ValueError("n_sectors must be >= 1")
        if self.alpha_reference_active_gross <= 0:
            raise ValueError("alpha_reference_active_gross must be positive")
        if not 0 <= self.signal_autocorr < 1:
            raise ValueError("signal_autocorr must be in [0, 1)")
        for name in (
            "market_vol",
            "sector_vol",
            "idio_vol",
            "dividend_yield",
            "management_fee",
            "borrow_cost",
            "transaction_cost",
            "debit_rate",
            "cash_rate",
            "harvest_threshold",
            "rebalance_band",
            "ordinary_offset_limit",
            "outside_st_gains_annual",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("long_maintenance", "short_maintenance"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if not 0 < self.deleverage_buffer <= 1:
            raise ValueError("deleverage_buffer must be in (0, 1]")
        if not 0 <= self.harvest_exposure_floor <= 1:
            raise ValueError("harvest_exposure_floor must be in [0, 1]")
        import math

        for field_name, value in vars(self).items():
            if isinstance(value, int | float) and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite, got {value!r}")
        # The event schedule is a dict, so the scalar checks above never see
        # its contents: validate every key and value explicitly.
        for year, amount in self.outside_st_gain_events.items():
            if isinstance(year, bool) or not isinstance(year, int):
                raise ValueError(f"outside_st_gain_events keys must be int years, got {year!r}")
            if not 0 <= year < self.years:
                raise ValueError(
                    f"outside_st_gain_events year {year} is outside the horizon [0, {self.years})"
                )
            if isinstance(amount, bool) or not isinstance(amount, int | float):
                raise ValueError(f"outside_st_gain_events[{year}] must be a number, got {amount!r}")
            if not math.isfinite(amount):
                raise ValueError(f"outside_st_gain_events[{year}] must be finite, got {amount!r}")
            if amount < 0:
                raise ValueError(f"outside_st_gain_events[{year}] must be non-negative")
        if self.margin_response == "deleverage":
            core_req = self.long_maintenance * (self.long_exposure - self.short_exposure)
            if core_req > self.deleverage_buffer:
                raise ValueError(
                    "net core exposure alone violates the maintenance requirement; "
                    "deleveraging only shrinks the long/short extension, so this "
                    "book cannot be made compliant. Reduce net exposure or use "
                    "margin_response='flag'."
                )

    @property
    def gross_exposure(self) -> float:
        return self.long_exposure + self.short_exposure

    @property
    def active_gross(self) -> float:
        """Gross active exposure vs a fully invested long-only benchmark."""
        return (self.long_exposure - 1.0) + self.short_exposure

    @property
    def n_steps(self) -> int:
        return self.years * self.steps_per_year

    def with_book(self, name: str) -> ScenarioConfig:
        """Return a copy of this config with a preset long/short book."""
        long_exp, short_exp = BOOK_PRESETS[name]
        from dataclasses import replace

        return replace(self, long_exposure=long_exp, short_exposure=short_exp)

    def outside_st_gain_for_year(self, year: int) -> float:
        return self.outside_st_gains_annual + self.outside_st_gain_events.get(year, 0.0)
