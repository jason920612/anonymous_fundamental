"""Phase 36 — SEC + price cache freshness checks."""

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from afp.data.cache_refresh import (
    _latest_accepted,
    refresh_prices_for_ticker,
    refresh_sec_for_cik,
)


def _stub_submissions_payload(accepted: list[str], files: list | None = None):
    return {
        "cik": "0000320193",
        "filings": {
            "recent": {
                "accessionNumber": ["A"] * len(accepted),
                "form": ["10-Q"] * len(accepted),
                "filingDate": accepted,
                "reportDate": accepted,
                "acceptanceDateTime": accepted,
                "primaryDocument": ["doc.htm"] * len(accepted),
                "isXBRL": [1] * len(accepted),
                "isInlineXBRL": [1] * len(accepted),
            },
            "files": files or [],
        },
    }


def test_latest_accepted_extracts_max():
    payload = _stub_submissions_payload([
        "2024-01-15T16:30:00", "2024-04-20T16:00:00", "2024-07-10T17:00:00"])
    assert _latest_accepted(payload) == "2024-07-10T17:00:00"


def test_refresh_sec_writes_when_cache_is_stale(tmp_path):
    cik = "0000320193"
    sub_dir = tmp_path / "submissions"; sub_dir.mkdir()
    facts_dir = tmp_path / "companyfacts"; facts_dir.mkdir()
    # Pre-populate stale cache
    (sub_dir / f"CIK{cik}.json").write_text(json.dumps(
        _stub_submissions_payload(["2024-01-15T16:30:00"])))
    (facts_dir / f"CIK{cik}.json").write_text(json.dumps({"cik": cik, "facts": {}}))

    sec = MagicMock()
    sec.submissions = MagicMock(return_value=_stub_submissions_payload([
        "2024-01-15T16:30:00", "2024-04-20T16:00:00"]))
    sec.get_json = MagicMock(side_effect=Exception)
    sec.company_facts = MagicMock(return_value={"cik": cik, "facts": {"updated": True}})

    pages, facts, refreshed = refresh_sec_for_cik(cik, sec, tmp_path)
    assert refreshed is True
    assert sec.company_facts.called
    assert _latest_accepted(pages[0]) == "2024-04-20T16:00:00"


def test_refresh_sec_no_op_when_cache_matches(tmp_path):
    cik = "0000320193"
    sub_dir = tmp_path / "submissions"; sub_dir.mkdir()
    facts_dir = tmp_path / "companyfacts"; facts_dir.mkdir()
    payload = _stub_submissions_payload(["2024-04-20T16:00:00"])
    (sub_dir / f"CIK{cik}.json").write_text(json.dumps(payload))
    (facts_dir / f"CIK{cik}.json").write_text(json.dumps({"cik": cik, "facts": {}}))

    sec = MagicMock()
    sec.submissions = MagicMock(return_value=payload)
    sec.company_facts = MagicMock(return_value={"cik": cik, "facts": {}})
    sec.get_json = MagicMock(side_effect=Exception)
    pages, facts, refreshed = refresh_sec_for_cik(cik, sec, tmp_path)
    assert refreshed is False
    assert not sec.company_facts.called    # facts not re-downloaded


def test_refresh_prices_uses_cache_when_fresh(tmp_path):
    cache = tmp_path / "AAPL.parquet"
    today = date.today()
    df = pd.DataFrame({
        "date": [today - timedelta(days=i) for i in range(5)][::-1],
        "open": 100, "high": 100, "low": 100, "close": 100,
        "adjusted_close": 100, "volume": 1_000_000,
    })
    df.to_parquet(cache, index=False)

    yf = MagicMock()
    yf.get_daily_prices = MagicMock(return_value=pd.DataFrame())
    out, refreshed = refresh_prices_for_ticker(
        "AAPL", yf, str(today - timedelta(days=30)), str(today), tmp_path,
        max_age_trading_days=2)
    assert refreshed is False
    assert not yf.get_daily_prices.called
    assert len(out) == 5


def test_refresh_prices_fetches_when_stale(tmp_path):
    cache = tmp_path / "AAPL.parquet"
    today = date.today()
    df = pd.DataFrame({
        "date": [today - timedelta(days=30)],
        "open": 100, "high": 100, "low": 100, "close": 100,
        "adjusted_close": 100, "volume": 1_000_000,
    })
    df.to_parquet(cache, index=False)

    yf = MagicMock()
    yf.get_daily_prices = MagicMock(return_value=pd.DataFrame({
        "date": [today], "open": 100, "high": 100, "low": 100, "close": 100,
        "adjusted_close": 100, "volume": 1_000_000,
    }))
    out, refreshed = refresh_prices_for_ticker(
        "AAPL", yf, str(today - timedelta(days=60)), str(today), tmp_path,
        max_age_trading_days=2)
    assert refreshed is True
    assert yf.get_daily_prices.called
