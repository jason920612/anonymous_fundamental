"""yfinance-backed PriceClient with on-disk caching and rate limiting."""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd

from afp.data.price_client import REQUIRED_PRICE_COLUMNS
from afp.utils.logging import get_logger

log = get_logger(__name__)


class YFinancePriceClient:
    """Pulls daily OHLCV+adjusted_close from Yahoo Finance via the `yfinance` package.

    Each ticker is cached on disk so re-runs are cheap and we stay polite to the API.
    Falls back to the cached file if a fresh fetch fails.
    """

    def __init__(self, cache_root: str | Path = "data/raw/prices_cache",
                 pause_seconds: float = 0.5):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.pause_seconds = pause_seconds
        self._last_request = 0.0

    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        delta = time.monotonic() - self._last_request
        if delta < self.pause_seconds:
            time.sleep(self.pause_seconds - delta)
        self._last_request = time.monotonic()

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_root / f"{ticker.upper()}.parquet"

    # ------------------------------------------------------------------

    def _fetch(self, ticker: str, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance not installed — `pip install 'anonymous-fundamental-portfolio[prices]'`") from exc

        self._throttle()
        raw = yf.download(ticker, start=str(start_date), end=str(end_date),
                          auto_adjust=False, progress=False, threads=False)
        if raw is None or raw.empty:
            return pd.DataFrame(columns=REQUIRED_PRICE_COLUMNS)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        raw = raw.reset_index().rename(columns={"Date": "date", "Open": "open",
                                                "High": "high", "Low": "low",
                                                "Close": "close",
                                                "Adj Close": "adjusted_close",
                                                "Volume": "volume"})
        if "adjusted_close" not in raw.columns:
            # auto_adjust=True path
            raw["adjusted_close"] = raw["close"]
        raw["date"] = pd.to_datetime(raw["date"]).dt.date
        for col in REQUIRED_PRICE_COLUMNS:
            if col not in raw.columns:
                raw[col] = float("nan")
        raw = raw[list(REQUIRED_PRICE_COLUMNS)]
        return raw

    def get_daily_prices(self, ticker: str, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        ticker = ticker.upper()
        cache = self._cache_path(ticker)
        try:
            df = self._fetch(ticker, start_date, end_date)
            if not df.empty:
                df.to_parquet(cache, index=False)
            else:
                log.warning("yfinance_empty", extra={"ticker": ticker})
        except Exception as exc:
            log.warning("yfinance_fetch_failed", extra={"ticker": ticker, "err": str(exc)})
            if cache.exists():
                df = pd.read_parquet(cache)
            else:
                raise
        # Filter to requested window (cache may contain a wider range)
        start = pd.Timestamp(start_date).date()
        end = pd.Timestamp(end_date).date()
        if not df.empty:
            mask = (df["date"] >= start) & (df["date"] <= end)
            df = df.loc[mask].reset_index(drop=True)
        return df
