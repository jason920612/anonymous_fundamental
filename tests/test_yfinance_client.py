"""Phase 10 tests — yfinance client (mocked) + cache + fallback semantics."""

from unittest.mock import patch

import pandas as pd
import pytest

from afp.data.price_client_yfinance import YFinancePriceClient
from afp.data.price_client import REQUIRED_PRICE_COLUMNS


def _fake_payload():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame({
        "Date": idx, "Open": 100.0, "High": 101.0, "Low": 99.0,
        "Close": 100.5, "Adj Close": 100.5, "Volume": 1_000_000,
    })


def test_cache_round_trip(tmp_path):
    cli = YFinancePriceClient(cache_root=tmp_path, pause_seconds=0)
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3, freq="B").date,
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
        "adjusted_close": 1.0, "volume": 10,
    })
    df.to_parquet(cli._cache_path("XYZ"), index=False)
    with patch("afp.data.price_client_yfinance.YFinancePriceClient._fetch",
               side_effect=RuntimeError("offline")):
        out = cli.get_daily_prices("XYZ", "2024-01-01", "2024-01-05")
    assert len(out) == 3
    assert set(REQUIRED_PRICE_COLUMNS) <= set(out.columns)


def test_fetch_normalizes_columns(tmp_path):
    cli = YFinancePriceClient(cache_root=tmp_path, pause_seconds=0)
    with patch("afp.data.price_client_yfinance.YFinancePriceClient._fetch",
               return_value=pd.DataFrame({
                   "date": pd.date_range("2024-01-01", periods=5, freq="B").date,
                   "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                   "adjusted_close": 1.0, "volume": 10,
               })):
        out = cli.get_daily_prices("ABC", "2024-01-01", "2024-01-08")
    assert len(out) == 5
    assert list(out.columns) == list(REQUIRED_PRICE_COLUMNS) or \
           set(REQUIRED_PRICE_COLUMNS) <= set(out.columns)


def test_empty_fetch_returns_empty(tmp_path):
    cli = YFinancePriceClient(cache_root=tmp_path, pause_seconds=0)
    with patch("afp.data.price_client_yfinance.YFinancePriceClient._fetch",
               return_value=pd.DataFrame(columns=list(REQUIRED_PRICE_COLUMNS))):
        out = cli.get_daily_prices("NONE", "2024-01-01", "2024-01-08")
    assert out.empty
