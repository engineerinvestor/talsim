import pytest

from talsim.tax import settle_year

RATES = {"st_rate": 0.408, "lt_rate": 0.238, "ordinary_rate": 0.408}


def test_outside_gains_alone_pay_baseline_tax():
    r = settle_year(0, 0, 100_000, 0, 0, **RATES)
    assert r.capital_tax == pytest.approx(100_000 * 0.408)
    assert r.benefit_used == pytest.approx(0.0)
    assert r.carry_st == 0 and r.carry_lt == 0


def test_portfolio_losses_offset_outside_gains():
    r = settle_year(-40_000, 0, 100_000, 0, 0, **RATES)
    assert r.capital_tax == pytest.approx(60_000 * 0.408)
    assert r.benefit_used == pytest.approx(40_000 * 0.408)


def test_st_losses_offset_lt_gains_cross_character():
    r = settle_year(-50_000, 30_000, 0, 0, 0, **RATES)
    assert r.net_lt == pytest.approx(0.0)
    assert r.ordinary_offset == pytest.approx(3_000)
    assert r.carry_st == pytest.approx(17_000)
    assert r.benefit_used == pytest.approx(3_000 * 0.408)


def test_ordinary_offset_capped_at_limit():
    r = settle_year(-500_000, 0, 0, 0, 0, **RATES)
    assert r.ordinary_offset == pytest.approx(3_000)
    assert r.carry_st == pytest.approx(497_000)


def test_carryforward_used_next_year():
    y1 = settle_year(-200_000, 0, 100_000, 0, 0, **RATES)
    assert y1.capital_tax == 0
    assert y1.carry_st == pytest.approx(97_000)
    y2 = settle_year(0, 0, 100_000, y1.carry_st, y1.carry_lt, **RATES)
    assert y2.capital_tax == pytest.approx(3_000 * 0.408)
    assert y2.benefit_used == pytest.approx(97_000 * 0.408)


def test_net_gains_pay_more_than_baseline_and_benefit_is_zero():
    r = settle_year(50_000, 20_000, 100_000, 0, 0, **RATES)
    assert r.capital_tax == pytest.approx(150_000 * 0.408 + 20_000 * 0.238)
    assert r.benefit_used == 0.0


def test_negative_carry_rejected():
    with pytest.raises(ValueError):
        settle_year(0, 0, 0, -1, 0, **RATES)


# ---------------------------------------------------------------------------
# Dividend buckets (review blocker 3 regression tests)
# ---------------------------------------------------------------------------


def test_capital_losses_cannot_absorb_qualified_dividends():
    """$100k of qualified dividends must be taxed at the preferential rate
    in full, even alongside a $50k capital loss; only the $3k ordinary
    offset escapes."""
    r = settle_year(0, -50_000, 0, 0, 0, **RATES, qualified_dividends=100_000)
    assert r.dividend_tax == pytest.approx(100_000 * 0.238)
    assert r.ordinary_offset == pytest.approx(3_000)
    assert r.carry_lt == pytest.approx(47_000)


def test_ordinary_dividends_taxed_at_ordinary_rate():
    r = settle_year(0, 0, 0, 0, 0, **RATES, ordinary_dividends=10_000)
    assert r.dividend_tax == pytest.approx(10_000 * 0.408)
    assert r.capital_tax == 0.0


def test_dividends_never_negative():
    with pytest.raises(ValueError):
        settle_year(0, 0, 0, 0, 0, **RATES, qualified_dividends=-1)


# ---------------------------------------------------------------------------
# Liquidation-tax definition (review blocker 4 regression test)
# ---------------------------------------------------------------------------


def test_liquidation_tax_counterexample_from_review():
    """$100k pre-liquidation ST loss offsets $100k outside gains; the
    liquidation adds a $100k LT gain. Incremental household tax must be
    $23,800, not $47,600."""
    cf = settle_year(-100_000, 0, 100_000, 0, 0, **RATES)
    fin = settle_year(-100_000, 100_000, 100_000, 0, 0, **RATES)
    t_without = cf.household_tax - cf.ordinary_offset * RATES["ordinary_rate"]
    t_with = fin.household_tax - fin.ordinary_offset * RATES["ordinary_rate"]
    assert t_with - t_without == pytest.approx(23_800.0)
