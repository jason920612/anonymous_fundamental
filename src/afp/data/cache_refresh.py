"""Phase 36: keep on-disk SEC + price caches fresh.

Per-CIK and per-ticker checks that compare what's on disk to what's live,
re-fetching only when the source has new data. Avoids the "predict using
stale data" failure mode.

Cheap design:
  - SEC submissions: small JSON (<100KB). Always fetch fresh, compare
    `accepted_datetime` lists, replace cache if new filings appeared.
  - SEC companyfacts: bigger (100KB-1MB). Refresh only when submissions
    revealed a new accepted filing.
  - yfinance prices: read cache parquet, check max date; refresh if older
    than 2 trading days (today + yesterday window).
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Tuple

import pandas as pd

from afp.data.sec_client import PermanentSecError, SecClient
from afp.data.price_client_yfinance import YFinancePriceClient
from afp.utils.logging import get_logger

log = get_logger(__name__)


def _latest_accepted(sub_payload: dict) -> str | None:
    accepted = (sub_payload.get("filings") or {}).get("recent", {}).get("acceptanceDateTime") or []
    if not accepted:
        return None
    return max(accepted)


def refresh_sec_for_cik(cik: str, sec: SecClient, raw_dir: Path) -> tuple[dict | None, dict | None, bool]:
    """Return `(submissions_payload, facts_payload, refreshed)`.

    submissions_payload is always fresh (fetched). facts_payload is fresh
    only if new filings appeared since cache or cache was missing.
    `refreshed=True` means at least one disk write happened.
    """
    sub_cache = raw_dir / "submissions" / f"CIK{cik}.json"
    facts_cache = raw_dir / "companyfacts" / f"CIK{cik}.json"
    failed_marker = raw_dir / "companyfacts" / "_404_ciks.json"
    sub_cache.parent.mkdir(parents=True, exist_ok=True)
    facts_cache.parent.mkdir(parents=True, exist_ok=True)
    known_404: set[str] = set()
    if failed_marker.exists():
        try:
            known_404 = set(json.loads(failed_marker.read_text()))
        except Exception:
            known_404 = set()
    if cik in known_404:
        return None, None, False

    fresh = sec.submissions(cik)
    fresh_latest = _latest_accepted(fresh)
    refreshed = False

    cached_sub = None
    if sub_cache.exists():
        try:
            cached_sub = json.loads(sub_cache.read_text())
        except Exception:
            cached_sub = None
    cached_latest = _latest_accepted(cached_sub) if cached_sub else None

    if cached_latest != fresh_latest:
        sub_cache.write_text(json.dumps(fresh))
        refreshed = True
        log.info("sec_submissions_refreshed",
                 extra={"cik": cik, "from": cached_latest, "to": fresh_latest})

    # Walk pagination IF present
    pages = [fresh]
    for f in (fresh.get("filings") or {}).get("files") or []:
        name = f.get("name")
        if not name:
            continue
        try:
            page = sec.get_json(f"https://data.sec.gov/submissions/{name}")
            pages.append({"cik": cik, "filings": {"recent": page}})
        except Exception:
            continue

    # Facts cache: refresh if submissions advanced OR facts cache missing
    need_facts_refresh = refreshed or not facts_cache.exists()
    facts_payload = None
    if need_facts_refresh:
        try:
            facts_payload = sec.company_facts(cik)
            facts_cache.write_text(json.dumps(facts_payload))
            refreshed = True
        except PermanentSecError as exc:
            new_known = known_404 | {cik}
            failed_marker.write_text(json.dumps(sorted(new_known)))
            log.warning("sec_facts_404_marked", extra={"cik": cik, "status": exc.status_code})
            return pages, None, refreshed
    elif facts_cache.exists():
        try:
            facts_payload = json.loads(facts_cache.read_text())
        except Exception:
            facts_payload = sec.company_facts(cik)
            facts_cache.write_text(json.dumps(facts_payload))
            refreshed = True

    return pages, facts_payload, refreshed


def _price_cache_max_date(path: Path) -> date | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=["date"])
    except Exception:
        return None
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).max().date()


def refresh_prices_for_ticker(
    ticker: str,
    yf: YFinancePriceClient,
    start_date: str,
    end_date: str,
    cache_root: Path,
    max_age_trading_days: int = 2,
) -> tuple[pd.DataFrame, bool]:
    """Return `(prices_df, refreshed)`. Refreshes when cache's latest date
    is older than `today - max_age_trading_days` (so weekend / holiday gaps
    don't trigger unnecessary fetches).
    """
    cache = cache_root / f"{ticker.upper()}.parquet"
    today = date.today()
    cutoff = today - timedelta(days=max_age_trading_days + 4)  # allow weekend
    cached_max = _price_cache_max_date(cache)
    refreshed = False
    if cached_max is None or cached_max < cutoff:
        df = yf.get_daily_prices(ticker, start_date, end_date)
        refreshed = True
        log.info("prices_refreshed", extra={"ticker": ticker, "from_max": str(cached_max),
                                            "to_max": str(today)})
    else:
        # Read cache and slice
        df = pd.read_parquet(cache)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        mask = (df["date"] >= pd.Timestamp(start_date).date()) & \
               (df["date"] <= pd.Timestamp(end_date).date())
        df = df.loc[mask].reset_index(drop=True)
    return df, refreshed
