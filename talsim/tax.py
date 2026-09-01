"""Annual capital-gain netting, the ordinary-income offset, and carryforwards.

The netting order follows the federal scheme in simplified form: short-term
gains and losses net first within character, then net losses of one character
offset net gains of the other, then up to the ordinary-income limit
($3,000) of any remaining net loss offsets ordinary income, and the rest
carries forward preserving character.

`settle_year` also decomposes the household's tax into two reportable
quantities:

- `tax_paid`: tax actually owed this year on outside gains plus portfolio
  realizations (after carryforwards).
- `benefit_used`: the reduction in household tax versus a counterfactual
  with no portfolio activity and no carryforwards. This is the only number
  that deserves to be called a realized tax benefit; gross harvested losses
  and net losses that merely enter the carryforward are reported separately.

State tax is out of scope. Rates are injected, never hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class YearSettlement:
    tax_paid: float
    benefit_used: float
    carry_st: float  # positive numbers: losses carried forward
    carry_lt: float
    net_st: float  # post-netting taxable amounts (positive = taxable gain)
    net_lt: float
    ordinary_offset: float


def _net(st: float, lt: float) -> tuple[float, float]:
    """Cross-character netting: a net loss in one bucket offsets the other."""
    if st < 0 and lt > 0:
        applied = min(-st, lt)
        st += applied
        lt -= applied
    elif lt < 0 and st > 0:
        applied = min(-lt, st)
        lt += applied
        st -= applied
    return st, lt


def settle_year(
    portfolio_st: float,
    portfolio_lt: float,
    outside_st: float,
    carry_st: float,
    carry_lt: float,
    st_rate: float,
    lt_rate: float,
    ordinary_rate: float,
    ordinary_offset_limit: float = 3_000.0,
) -> YearSettlement:
    if carry_st < 0 or carry_lt < 0:
        raise ValueError("carryforwards are stored as non-negative loss amounts")
    if outside_st < 0:
        raise ValueError("outside gains must be non-negative in this model")

    # Actual household position: outside gains + portfolio + carried losses.
    st = outside_st + portfolio_st - carry_st
    lt = portfolio_lt - carry_lt
    st, lt = _net(st, lt)

    ordinary_offset = 0.0
    remaining_loss = -(min(st, 0.0) + min(lt, 0.0))
    if remaining_loss > 0:
        ordinary_offset = min(remaining_loss, ordinary_offset_limit)
        # The offset consumes short-term loss first (worst-for-taxpayer last).
        st_loss = -min(st, 0.0)
        take_st = min(st_loss, ordinary_offset)
        st += take_st
        lt += ordinary_offset - take_st

    tax_paid = max(st, 0.0) * st_rate + max(lt, 0.0) * lt_rate

    # Counterfactual: outside gains alone, no portfolio, no carryforwards.
    baseline_tax = outside_st * st_rate

    benefit_used = (baseline_tax - tax_paid) + ordinary_offset * ordinary_rate
    # A portfolio with net realized gains costs tax instead of saving it;
    # benefit_used is only the saving, never negative.
    benefit_used = max(benefit_used, 0.0)

    return YearSettlement(
        tax_paid=tax_paid,
        benefit_used=benefit_used,
        carry_st=-min(st, 0.0),
        carry_lt=-min(lt, 0.0),
        net_st=st,
        net_lt=lt,
        ordinary_offset=ordinary_offset,
    )
