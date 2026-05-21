"""S&P 500 constituent fetcher.

Source: Wikipedia maintains a canonical table of the current S&P 500
constituents at
https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

We pull it via pandas.read_html and cache the resulting parquet locally.
The cache TTL defaults to 24 hours since constituent changes are rare.

Output columns: ticker, name, sector, sub_industry, cik (zero-padded
10-digit string matching the SEC convention).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CACHE_PATH = Path("data/processed/sp500_constituents.parquet")


@dataclass
class SP500Config:
    cache_ttl_hours: int = 24
    user_agent: str = "anonymous-fundamental-portfolio research"


def _cache_is_fresh(cache_path: Path, ttl_hours: int) -> bool:
    if not cache_path.exists():
        return False
    age_seconds = time.time() - cache_path.stat().st_mtime
    return age_seconds < ttl_hours * 3600


def _fetch_from_wikipedia() -> pd.DataFrame:
    """Pull the constituents table from Wikipedia.

    Wikipedia's table has columns:
      Symbol, Security, GICS Sector, GICS Sub-Industry, Headquarters Location,
      Date added, CIK, Founded
    """
    # pandas.read_html accepts URLs and parses HTML tables. We grab the
    # first table on the page (constituents).
    tables = pd.read_html(WIKIPEDIA_URL, storage_options={"User-Agent": "Mozilla/5.0"})
    if not tables:
        raise RuntimeError("No tables found at Wikipedia S&P 500 page")
    df = tables[0]
    # Standardize column names
    rename_map = {
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
        "CIK": "cik",
    }
    df = df.rename(columns=rename_map)
    # Some columns may be missing if Wikipedia layout changes; keep what we have
    keep = [c for c in ["ticker", "name", "sector", "sub_industry", "cik"] if c in df.columns]
    df = df[keep].copy()
    # Format CIK as zero-padded 10-digit string (SEC convention)
    if "cik" in df.columns:
        df["cik"] = df["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    # Convert tickers like "BRK.B" → "BRK-B" (yfinance convention)
    df["ticker_yf"] = df["ticker"].astype(str).str.replace(".", "-", regex=False)
    return df.reset_index(drop=True)


def load_sp500_constituents(cfg: SP500Config | None = None,
                            force_refresh: bool = False) -> pd.DataFrame:
    """Load the current S&P 500 constituent list, caching to disk.

    Parameters
    ----------
    cfg : optional config (TTL, user agent).
    force_refresh : bypass the cache and re-fetch.

    Returns
    -------
    DataFrame with columns: ticker, name, sector, sub_industry, cik, ticker_yf.
    """
    cfg = cfg or SP500Config()
    if not force_refresh and _cache_is_fresh(CACHE_PATH, cfg.cache_ttl_hours):
        return pd.read_parquet(CACHE_PATH)
    df = _fetch_from_wikipedia()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    return df
