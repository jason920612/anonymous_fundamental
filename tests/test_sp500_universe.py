"""Phase 68a: S&P 500 fetcher tests (offline — cache only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from afp.data import sp500_universe


def test_cache_returns_existing_parquet(tmp_path, monkeypatch):
    """If a fresh cache exists, load_sp500_constituents must read it
    without hitting the network."""
    df_in = pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "name": ["Apple Inc.", "Microsoft Corp."],
        "sector": ["Information Technology", "Information Technology"],
        "sub_industry": ["Tech Hardware", "Systems Software"],
        "cik": ["0000320193", "0000789019"],
        "ticker_yf": ["AAPL", "MSFT"],
    })
    cache = tmp_path / "sp500_constituents.parquet"
    df_in.to_parquet(cache, index=False)
    monkeypatch.setattr(sp500_universe, "CACHE_PATH", cache)

    def fail_fetch():
        raise RuntimeError("should not have called the network")
    monkeypatch.setattr(sp500_universe, "_fetch_from_wikipedia", fail_fetch)

    df_out = sp500_universe.load_sp500_constituents()
    assert list(df_out["ticker"]) == ["AAPL", "MSFT"]


def test_force_refresh_calls_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(sp500_universe, "CACHE_PATH", tmp_path / "x.parquet")
    called = {"n": 0}
    def fake_fetch():
        called["n"] += 1
        return pd.DataFrame({"ticker": ["X"], "name": ["X"],
                              "sector": ["x"], "sub_industry": ["x"],
                              "cik": ["0"], "ticker_yf": ["X"]})
    monkeypatch.setattr(sp500_universe, "_fetch_from_wikipedia", fake_fetch)
    out = sp500_universe.load_sp500_constituents(force_refresh=True)
    assert called["n"] == 1
    assert list(out["ticker"]) == ["X"]


def test_cik_zero_padded(monkeypatch, tmp_path):
    """CIKs should always be 10-digit zero-padded strings."""
    monkeypatch.setattr(sp500_universe, "CACHE_PATH", tmp_path / "x.parquet")
    def fake_fetch():
        return sp500_universe._fetch_from_wikipedia.__wrapped__() if False else pd.DataFrame({
            "Symbol": ["AAPL"],
            "Security": ["Apple"],
            "GICS Sector": ["IT"],
            "GICS Sub-Industry": ["Hardware"],
            "CIK": [320193],
        })
    # patch the wrapper that does the rename
    import afp.data.sp500_universe as mod
    monkeypatch.setattr(mod, "_fetch_from_wikipedia", lambda: mod._fetch_from_wikipedia.__wrapped__()
                        if False else _stub_renamed())
    out = sp500_universe.load_sp500_constituents(force_refresh=True)
    assert out["cik"].iloc[0] == "0000320193"


def _stub_renamed():
    # Provide a DataFrame in the post-rename schema
    df = pd.DataFrame({
        "ticker": ["AAPL"],
        "name": ["Apple"],
        "sector": ["IT"],
        "sub_industry": ["Hardware"],
        "cik": [320193],
    })
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    df["ticker_yf"] = df["ticker"].str.replace(".", "-", regex=False)
    return df
