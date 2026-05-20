"""US equity trading calendar.

RFC-01 §2.6 requires next/previous trading day lookups. We use NYSE rules:
- weekdays only (Mon-Fri)
- exclude the standard US market holidays observed by NYSE/Nasdaq

The exact holiday set is generated deterministically from rule helpers so the
calendar is reproducible without external data dependencies. For research use
the conservative approximation is sufficient; for live trading you would swap in
`pandas_market_calendars`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import pandas as pd


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """`n`-th occurrence of `weekday` (Mon=0..Sun=6) in `month`."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Last occurrence of `weekday` in `month`."""
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset)


def _observed(d: date) -> date:
    """If a fixed-date holiday falls on Sat/Sun, NYSE observes Fri/Mon."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _easter(year: int) -> date:
    """Anonymous Gregorian Easter algorithm — for Good Friday calculation."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    L = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * L) // 451
    month = (h + L - 7 * m + 114) // 31
    day = ((h + L - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def us_market_holidays(year: int) -> set[date]:
    """Approximate NYSE holiday set for a given year."""
    h: set[date] = set()
    h.add(_observed(date(year, 1, 1)))                                # New Year's
    h.add(_nth_weekday(year, 1, 0, 3))                                # MLK Day (3rd Mon Jan)
    h.add(_nth_weekday(year, 2, 0, 3))                                # Washington Birthday
    h.add(_easter(year) - timedelta(days=2))                          # Good Friday
    h.add(_last_weekday(year, 5, 0))                                  # Memorial Day
    if year >= 2021:
        h.add(_observed(date(year, 6, 19)))                           # Juneteenth
    h.add(_observed(date(year, 7, 4)))                                # Independence Day
    h.add(_nth_weekday(year, 9, 0, 1))                                # Labor Day
    h.add(_nth_weekday(year, 11, 3, 4))                               # Thanksgiving
    h.add(_observed(date(year, 12, 25)))                              # Christmas
    return h


@dataclass
class TradingCalendar:
    """Sorted array of trading dates with O(log n) lookup."""

    trading_dates: pd.DatetimeIndex

    @classmethod
    def build(cls, start: str | date, end: str | date) -> "TradingCalendar":
        start = pd.Timestamp(start).date()
        end = pd.Timestamp(end).date()
        if end < start:
            raise ValueError("end before start")

        holidays: set[date] = set()
        for year in range(start.year, end.year + 1):
            holidays |= us_market_holidays(year)

        all_days = pd.bdate_range(start, end).date
        trading = [d for d in all_days if d not in holidays]
        return cls(trading_dates=pd.DatetimeIndex(trading))

    # ---- Lookup helpers (RFC-01 §2.6) ----

    def is_trading_day(self, d: date) -> bool:
        return pd.Timestamp(d) in self.trading_dates

    def next_trading_day(self, d: date) -> date:
        ts = pd.Timestamp(d)
        idx = self.trading_dates.searchsorted(ts, side="right")
        if idx >= len(self.trading_dates):
            raise IndexError(f"no trading day after {d} (calendar ends {self.trading_dates[-1].date()})")
        return self.trading_dates[idx].date()

    def previous_trading_day(self, d: date) -> date:
        ts = pd.Timestamp(d)
        idx = self.trading_dates.searchsorted(ts, side="left")
        if idx <= 0:
            raise IndexError(f"no trading day before {d} (calendar starts {self.trading_dates[0].date()})")
        return self.trading_dates[idx - 1].date()

    def trading_days_between(self, start: date, end: date) -> int:
        """Inclusive count of trading days between `start` and `end`."""
        a = self.trading_dates.searchsorted(pd.Timestamp(start), side="left")
        b = self.trading_dates.searchsorted(pd.Timestamp(end), side="right")
        return int(b - a)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"date": self.trading_dates.date, "is_trading_day": True})

    @classmethod
    def from_dates(cls, dates: Iterable[date]) -> "TradingCalendar":
        idx = pd.DatetimeIndex(sorted({pd.Timestamp(d) for d in dates}))
        return cls(trading_dates=idx)
