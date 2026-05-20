"""Synthetic fixtures shared by phase 2..8 tests.

We synthesize a tiny but realistic SEC-shaped world:
  - 4 companies, 24 quarterly 10-Q/10-K filings each spanning 6 years
  - geometric Brownian price paths
  - a small XBRL concept catalogue with semi-random values
This avoids any network access during tests while exercising the real parsers
and dataset/feature builders.
"""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from afp.data.identifiers import internal_company_id, normalize_cik
from afp.data.trading_calendar import TradingCalendar

CONCEPTS = [
    ("us-gaap", "Revenues", "USD"),
    ("us-gaap", "NetIncomeLoss", "USD"),
    ("us-gaap", "Assets", "USD"),
    ("us-gaap", "Liabilities", "USD"),
    ("us-gaap", "CashAndCashEquivalentsAtCarryingValue", "USD"),
    ("us-gaap", "OperatingCashFlow", "USD"),
    ("us-gaap", "PropertyPlantAndEquipmentNet", "USD"),
    ("us-gaap", "ResearchAndDevelopmentExpense", "USD"),
    ("us-gaap", "InventoryNet", "USD"),
    ("us-gaap", "AccountsReceivableNetCurrent", "USD"),
]

DEFAULT_COMPANIES: list[dict[str, Any]] = [
    {"cik": 1, "ticker": "AAA", "name": "Alpha Industries"},
    {"cik": 2, "ticker": "BBB", "name": "Beta Holdings"},
    {"cik": 3, "ticker": "CCC", "name": "Gamma Manufacturing"},
    {"cik": 4, "ticker": "DDD", "name": "Delta Logistics"},
]


def make_company_tickers_json(companies: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    companies = companies or DEFAULT_COMPANIES
    return {str(i): {"cik_str": c["cik"], "ticker": c["ticker"], "title": c["name"]}
            for i, c in enumerate(companies)}


def _quarter_end(year: int, quarter: int) -> date:
    last_months = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    m, d = last_months[quarter]
    return date(year, m, d)


def make_submissions_json(cik: int, start_year: int = 2010, end_year: int = 2018) -> dict[str, Any]:
    cik10 = normalize_cik(cik)
    forms, accs, fdates, rdates, accepted, prim = [], [], [], [], [], []
    seq = 0
    for year in range(end_year, start_year - 1, -1):
        for q in (4, 3, 2, 1):
            seq += 1
            period_end = _quarter_end(year, q)
            filed = period_end + timedelta(days=35)
            accept = filed.isoformat() + "T18:30:00.000Z"
            accession = f"{cik:010d}-{year:04d}-{q:06d}"
            form = "10-K" if q == 4 else "10-Q"
            accs.append(accession)
            forms.append(form)
            fdates.append(filed.isoformat())
            rdates.append(period_end.isoformat())
            accepted.append(accept)
            prim.append(f"{accession}.htm")

    return {
        "cik": cik10,
        "filings": {"recent": {
            "accessionNumber": accs,
            "form": forms,
            "filingDate": fdates,
            "reportDate": rdates,
            "acceptanceDateTime": accepted,
            "primaryDocument": prim,
            "isXBRL": [1] * len(accs),
            "isInlineXBRL": [1] * len(accs),
        }},
    }


def _rng_for(cik: int) -> np.random.Generator:
    seed = int(hashlib.sha1(f"cik-{cik}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def make_company_facts_json(cik: int, start_year: int = 2010, end_year: int = 2018) -> dict[str, Any]:
    cik10 = normalize_cik(cik)
    rng = _rng_for(cik)
    facts: dict[str, dict[str, dict[str, dict[str, list[dict[str, Any]]]]]] = {}
    base_levels = {c[1]: float(rng.uniform(1e7, 1e10)) for c in CONCEPTS}
    drift = {c[1]: float(rng.normal(0.02, 0.05)) for c in CONCEPTS}

    for taxonomy, concept, unit in CONCEPTS:
        facts.setdefault(taxonomy, {}).setdefault(concept, {"units": {unit: []}})
        obs_list = facts[taxonomy][concept]["units"][unit]
        t = 0
        for year in range(start_year, end_year + 1):
            for q in (1, 2, 3, 4):
                period_end = _quarter_end(year, q)
                period_start = _quarter_end(year - (1 if q == 1 else 0), 4 if q == 1 else q - 1) + timedelta(days=1)
                filed = period_end + timedelta(days=35)
                value = base_levels[concept] * (1 + drift[concept]) ** t * rng.normal(1.0, 0.05)
                if concept in ("NetIncomeLoss", "OperatingCashFlow") and rng.random() < 0.1:
                    value = -value * rng.uniform(0.05, 0.4)
                obs_list.append({
                    "start": period_start.isoformat(),
                    "end": period_end.isoformat(),
                    "val": float(value),
                    "accn": f"{cik:010d}-{year:04d}-{q:06d}",
                    "fy": year,
                    "fp": f"Q{q}" if q != 4 else "FY",
                    "form": "10-K" if q == 4 else "10-Q",
                    "filed": filed.isoformat(),
                })
                t += 1
    return {"cik": cik10, "entityName": f"CIK{cik10}", "facts": facts}


def make_synthetic_prices(
    tickers: list[str],
    start: date,
    end: date,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Geometric Brownian motion price series on the (synthetic) trading calendar."""
    cal = TradingCalendar.build(start, end)
    dates = [d.date() for d in cal.trading_dates]
    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        n = len(dates)
        mu = rng.uniform(0.05, 0.20) / 252
        sigma = rng.uniform(0.15, 0.45) / np.sqrt(252)
        shocks = rng.normal(mu, sigma, n)
        log_price = np.cumsum(shocks) + np.log(rng.uniform(20, 200))
        close = np.exp(log_price)
        vol = rng.integers(500_000, 5_000_000, n)
        df = pd.DataFrame({
            "date": dates,
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "close": close,
            "adjusted_close": close,
            "volume": vol,
            "dividend": 0.0,
            "split_coefficient": 1.0,
            "source": "synthetic",
            "downloaded_at": "1970-01-01T00:00:00Z",
        })
        df["internal_company_id"] = internal_company_id(_ticker_to_cik(t))
        df["ticker_at_date"] = t
        out[t] = df
    return out


def _ticker_to_cik(ticker: str) -> int:
    mapping = {c["ticker"]: c["cik"] for c in DEFAULT_COMPANIES}
    if ticker in mapping:
        return mapping[ticker]
    # Synthetic benchmarks
    return {"SPY": 9001, "QQQ": 9002}.get(ticker, 9999)


def make_synthetic_world(start_year: int = 2010, end_year: int = 2018):
    """Convenience: build all parsed dataframes + price store for downstream tests."""
    from afp.data.parse_companies import parse_company_tickers
    from afp.data.parse_companyfacts import parse_company_facts
    from afp.data.parse_submissions import parse_submissions

    companies = parse_company_tickers(make_company_tickers_json())
    filings_frames, facts_frames = [], []
    for c in DEFAULT_COMPANIES:
        filings_frames.append(parse_submissions(
            make_submissions_json(c["cik"], start_year, end_year)))
        facts_frames.append(parse_company_facts(
            make_company_facts_json(c["cik"], start_year, end_year)))
    filings = pd.concat(filings_frames, ignore_index=True)
    facts = pd.concat(facts_frames, ignore_index=True)

    start = date(start_year, 1, 1)
    end = date(end_year + 1, 6, 30)
    cal = TradingCalendar.build(start, end)

    price_map = make_synthetic_prices(
        [c["ticker"] for c in DEFAULT_COMPANIES] + ["SPY", "QQQ"],
        start, end,
    )
    from afp.data.price_client import stack_price_frames
    prices = stack_price_frames(price_map.items())

    benchmarks = prices[prices["ticker_at_date"].isin(["SPY", "QQQ"])].copy()
    company_prices = prices[~prices["ticker_at_date"].isin(["SPY", "QQQ"])].copy()

    return {
        "companies": companies,
        "filings": filings,
        "facts": facts,
        "prices": company_prices,
        "benchmarks": benchmarks,
        "calendar": cal,
    }
