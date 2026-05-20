"""Price client interface — source-agnostic, returns adjusted close data."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd

REQUIRED_PRICE_COLUMNS = ("date", "open", "high", "low", "close", "adjusted_close", "volume")


class PriceClient(Protocol):
    """RFC-01 §3.2: pluggable source returning canonical OHLCV+adjclose frames."""

    def get_daily_prices(self, ticker: str, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        ...


class CsvPriceClient:
    """Reads pre-downloaded CSVs from `data/raw/prices/{TICKER}.csv`.

    Each CSV must contain at least the columns listed in `REQUIRED_PRICE_COLUMNS`.
    Useful for backtests on archived data and for unit tests.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def get_daily_prices(self, ticker: str, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        path = self.root / f"{ticker.upper()}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        missing = set(REQUIRED_PRICE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        df["date"] = pd.to_datetime(df["date"]).dt.date
        start = pd.Timestamp(start_date).date()
        end = pd.Timestamp(end_date).date()
        mask = (df["date"] >= start) & (df["date"] <= end)
        return df.loc[mask].reset_index(drop=True)


def validate_price_frame(df: pd.DataFrame, ticker: str) -> list[str]:
    """Cheap quality checks per RFC-01 §8.2. Returns list of warnings."""
    warnings: list[str] = []
    if df.empty:
        warnings.append(f"{ticker}: empty price frame")
        return warnings
    if df["date"].is_unique is False:
        warnings.append(f"{ticker}: duplicate dates")
    if (df["adjusted_close"] <= 0).any():
        warnings.append(f"{ticker}: non-positive adjusted_close present")
    if (df["volume"] < 0).any():
        warnings.append(f"{ticker}: negative volume present")
    return warnings


def stack_price_frames(frames: Iterable[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Concatenate per-ticker price frames into the canonical long table."""
    rows = []
    for ticker, df in frames:
        if df.empty:
            continue
        local = df.copy()
        local["ticker_at_date"] = ticker.upper()
        rows.append(local)
    if not rows:
        return pd.DataFrame(columns=list(REQUIRED_PRICE_COLUMNS) + ["ticker_at_date"])
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["ticker_at_date", "date"]).reset_index(drop=True)
