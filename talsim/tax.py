"""Annual settlement: capital-gain netting, dividend buckets, carryforwards.

The netting order follows the federal scheme in simplified form: gains and
losses net within character, then a net loss of one character offsets the
other, then up to the ordinary-income limit ($3,000) of any remaining net
loss offsets ordinary income, and the rest carries forward preserving
character.

Dividends are ordinary income, never capital gain. Qualified dividends are
taxed at the preferential (long-term) rate and non-qualified dividends at
the ordinary rate, but neither can be absorbed by capital losses beyond the
ordinary-income offset. They are therefore settled in their own buckets.

Investment interest expense (IRC 163(d)): payments in lieu on shorts held
more than 45 days and margin debit interest are itemized deductions limited
to net investment income, which here is interest income, non-qualified
dividends, and net short-term gain after netting (the household's outside
gains count). Qualified dividends and long-term gains count only under the
163(d)(4)(B) election, which is not modeled. Disallowed interest carries
forward without limit. Borrow fees and management fees are miscellaneous
itemized deductions, suspended since TCJA, and get no deduction.

`settle_year` decomposes the household's tax into reportable quantities:

- `capital_tax`: tax on net capital gains (outside plus portfolio, after
  carryforwards).
- `dividend_tax`: tax on the year's portfolio dividends.
- `investment_interest_saving`: ordinary tax saved by the interest deduction.
- `benefit_used`: the reduction in capital-gain tax versus a counterfactual
  with no portfolio and no carryforwards, plus the ordinary-offset saving.
  This is the only number that deserves to be called a realized benefit.

State tax is out of scope. Rates are injected, never hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class YearSettlement:
    capital_tax: float
    dividend_tax: float
    benefit_used: float
    carry_st: float  # positive numbers: losses carried forward
    carry_lt: float
    net_st: float  # post-netting taxable amounts (positive = taxable gain)
    net_lt: float
    ordinary_offset: float
    investment_interest_deducted: float = 0.0
    investment_interest_saving: float = 0.0
    carry_investment_interest: float = 0.0

    @property
    def household_tax(self) -> float:
        """Total tax net of the investment-interest saving but not of the
        ordinary-offset saving (which reduces tax on wage or other ordinary
        income outside this ledger)."""
        return self.capital_tax + self.dividend_tax - self.investment_interest_saving


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
    qualified_dividends: float = 0.0,
    ordinary_dividends: float = 0.0,
    investment_interest: float = 0.0,
    carry_investment_interest: float = 0.0,
    interest_income: float = 0.0,
) -> YearSettlement:
    if carry_st < 0 or carry_lt < 0:
        raise ValueError("carryforwards are stored as non-negative loss amounts")
    if outside_st < 0:
        raise ValueError("outside gains must be non-negative in this model")
    if qualified_dividends < 0 or ordinary_dividends < 0:
        raise ValueError("dividend buckets must be non-negative")
    if investment_interest < 0 or carry_investment_interest < 0 or interest_income < 0:
        raise ValueError("investment interest, its carryforward, and interest income must be >= 0")

    st = outside_st + portfolio_st - carry_st
    lt = portfolio_lt - carry_lt
    st, lt = _net(st, lt)

    ordinary_offset = 0.0
    remaining_loss = -(min(st, 0.0) + min(lt, 0.0))
    if remaining_loss > 0:
        ordinary_offset = min(remaining_loss, ordinary_offset_limit)
        st_loss = -min(st, 0.0)
        take_st = min(st_loss, ordinary_offset)
        st += take_st
        lt += ordinary_offset - take_st

    capital_tax = max(st, 0.0) * st_rate + max(lt, 0.0) * lt_rate
    dividend_tax = qualified_dividends * lt_rate + ordinary_dividends * ordinary_rate

    # Investment interest: deductible up to net investment income.
    net_investment_income = ordinary_dividends + interest_income + max(st, 0.0)
    available = investment_interest + carry_investment_interest
    deducted = min(available, net_investment_income)
    interest_saving = deducted * ordinary_rate

    # Counterfactual: outside gains alone, no portfolio, no carryforwards.
    baseline_capital_tax = outside_st * st_rate

    benefit_used = (baseline_capital_tax - capital_tax) + ordinary_offset * ordinary_rate
    benefit_used = max(benefit_used, 0.0)

    return YearSettlement(
        capital_tax=capital_tax,
        dividend_tax=dividend_tax,
        benefit_used=benefit_used,
        carry_st=-min(st, 0.0),
        carry_lt=-min(lt, 0.0),
        net_st=st,
        net_lt=lt,
        ordinary_offset=ordinary_offset,
        investment_interest_deducted=deducted,
        investment_interest_saving=interest_saving,
        carry_investment_interest=available - deducted,
    )
