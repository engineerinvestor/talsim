import pytest

from talsim.tax import settle_year

RATES = {"st_rate": 0.408, "lt_rate": 0.238, "ordinary_rate": 0.408}


def test_outside_gains_alone_pay_baseline_tax():
    r = settle_year(0, 0, 100_000, 0, 0, **RATES)
    assert r.tax_paid == pytest.approx(100_000 * 0.408)
    assert r.benefit_used == pytest.approx(0.0)
    assert r.carry_st == 0 and r.carry_lt == 0


def test_portfolio_losses_offset_outside_gains():
    r = settle_year(-40_000, 0, 100_000, 0, 0, **RATES)
    assert r.tax_paid == pytest.approx(60_000 * 0.408)
    assert r.benefit_used == pytest.approx(40_000 * 0.408)


def test_st_losses_offset_lt_gains_cross_character():
    r = settle_year(-50_000, 30_000, 0, 0, 0, **RATES)
    # 50k ST loss absorbs 30k LT gain; 20k loss left; 3k ordinary offset.
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
    assert y1.tax_paid == 0
    # Year 1 leaves 100k of loss, minus the 3k ordinary offset: 97k carries.
    assert y1.carry_st == pytest.approx(97_000)
    y2 = settle_year(0, 0, 100_000, y1.carry_st, y1.carry_lt, **RATES)
    assert y2.tax_paid == pytest.approx(3_000 * 0.408)
    assert y2.benefit_used == pytest.approx(97_000 * 0.408)


def test_net_gains_pay_more_than_baseline_and_benefit_is_zero():
    r = settle_year(50_000, 20_000, 100_000, 0, 0, **RATES)
    assert r.tax_paid == pytest.approx(150_000 * 0.408 + 20_000 * 0.238)
    assert r.benefit_used == 0.0


def test_negative_carry_rejected():
    with pytest.raises(ValueError):
        settle_year(0, 0, 0, -1, 0, **RATES)
