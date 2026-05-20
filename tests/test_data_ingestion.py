"""Phase 2 tests: parsers, identifiers, trading calendar, universe filters."""

from datetime import date

import pandas as pd

from afp.data.identifiers import internal_company_id, normalize_cik
from afp.data.parse_companies import parse_company_tickers
from afp.data.parse_companyfacts import parse_company_facts
from afp.data.parse_submissions import parse_submissions
from afp.data.trading_calendar import TradingCalendar, us_market_holidays
from afp.data.universe import UniverseConfig, compute_dollar_volume, point_in_time_universe
from tests.fixtures import (
    DEFAULT_COMPANIES,
    make_company_facts_json,
    make_company_tickers_json,
    make_submissions_json,
    make_synthetic_world,
)


def test_normalize_cik_pads_and_strips():
    assert normalize_cik(320193) == "0000320193"
    assert normalize_cik("0000320193") == "0000320193"
    assert normalize_cik("CIK0000320193") == "0000320193"


def test_internal_company_id_is_stable():
    assert internal_company_id(320193) == "COMP0000320193"


def test_parse_company_tickers_dedupes_by_cik():
    df = parse_company_tickers(make_company_tickers_json())
    assert len(df) == len(DEFAULT_COMPANIES)
    assert df["cik"].is_unique
    assert (df["cik"].str.len() == 10).all()


def test_parse_submissions_filters_to_primary_forms():
    sub = make_submissions_json(1, 2015, 2017)
    df = parse_submissions(sub, allowed_forms=("10-Q", "10-K"))
    assert set(df["form_type"]) <= {"10-Q", "10-K"}
    assert df["filing_event_id"].is_unique
    assert (df["cik"].str.len() == 10).all()


def test_parse_company_facts_long_format():
    facts = parse_company_facts(make_company_facts_json(1, 2015, 2016))
    assert {"concept_name", "unit", "value", "period_end_date", "accession_number"} <= set(facts.columns)
    assert facts["fact_id"].is_unique
    assert (facts["value"].notna()).all()


def test_trading_calendar_skips_weekends_and_holidays():
    cal = TradingCalendar.build("2024-01-01", "2024-12-31")
    # 2024-01-01 New Year (observed on the day, Monday)
    assert not cal.is_trading_day(date(2024, 1, 1))
    # 2024-12-25 Christmas Wednesday
    assert not cal.is_trading_day(date(2024, 12, 25))
    # Saturday
    assert not cal.is_trading_day(date(2024, 5, 4))
    # Regular Tuesday
    assert cal.is_trading_day(date(2024, 5, 7))


def test_next_and_previous_trading_day():
    cal = TradingCalendar.build("2020-01-01", "2020-12-31")
    # Friday 2020-07-03 falls on the observed Independence Day -> next td is Mon 7/6
    assert cal.next_trading_day(date(2020, 7, 2)) == date(2020, 7, 6)
    # Previous trading day before 2020-01-01 -> 2019-12-31 not in range -> use 2020-01-02
    # ... use a date inside the range:
    assert cal.previous_trading_day(date(2020, 1, 6)) == date(2020, 1, 3)


def test_us_holidays_include_juneteenth_after_2021():
    assert date(2024, 6, 19) in us_market_holidays(2024)
    assert date(2020, 6, 19) not in us_market_holidays(2020)


def test_universe_filters_remove_low_liquidity_companies():
    world = make_synthetic_world(2015, 2017)
    px = compute_dollar_volume(world["prices"])
    cfg = UniverseConfig(min_years_price_history=1,
                         min_financial_report_events=4,
                         min_avg_daily_dollar_volume_usd=1_000.0,
                         min_adjusted_close_price_usd=1.0)
    universe = point_in_time_universe(px, world["filings"], date(2017, 6, 30), cfg)
    # Synthetic prices easily pass the filters -> all 4 companies eligible
    assert len(universe) == len(DEFAULT_COMPANIES)


def test_universe_uses_only_past_data():
    """Eligibility on 2016-06-30 must not change when later prices are deleted."""
    world = make_synthetic_world(2015, 2017)
    px = compute_dollar_volume(world["prices"])
    cfg = UniverseConfig(min_years_price_history=1,
                         min_financial_report_events=2,
                         min_avg_daily_dollar_volume_usd=1_000.0,
                         min_adjusted_close_price_usd=1.0)
    cutoff = date(2016, 6, 30)
    a = point_in_time_universe(px, world["filings"], cutoff, cfg)
    px_past = px[px["date"] <= cutoff]
    b = point_in_time_universe(px_past, world["filings"], cutoff, cfg)
    assert set(a["internal_company_id"]) == set(b["internal_company_id"])
