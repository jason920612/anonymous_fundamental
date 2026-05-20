"""Parse SEC `company_tickers.json` into a normalized companies table."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from afp.data.identifiers import internal_company_id, normalize_cik


def parse_company_tickers(raw: dict[str, Any]) -> pd.DataFrame:
    """`raw` is the dict returned by SEC company_tickers.json.

    Schema follows RFC-01 §2.1.
    """
    rows = []
    iterator = raw.values() if isinstance(raw, dict) else raw
    for item in iterator:
        if not isinstance(item, dict):
            continue
        cik = item.get("cik_str") or item.get("cik")
        ticker = item.get("ticker")
        name = item.get("title") or item.get("name")
        if cik is None or ticker is None:
            continue
        cik10 = normalize_cik(cik)
        rows.append({
            "internal_company_id": internal_company_id(cik10),
            "cik": cik10,
            "ticker": str(ticker).upper(),
            "company_name": name,
            "exchange": item.get("exchange"),
            "security_type": "common_stock",
            "first_seen_date": None,
            "last_seen_date": None,
            "source": "sec_company_tickers",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["cik"]).reset_index(drop=True)
        df["downloaded_at"] = datetime.now(tz=timezone.utc).isoformat()
    return df
