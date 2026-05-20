"""Parse a single CIK's submissions.json into a filings DataFrame."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from afp.data.identifiers import internal_company_id, normalize_cik


def _filing_event_id(cik: str, accession: str) -> str:
    return "FILING_" + hashlib.sha1(f"{cik}|{accession}".encode()).hexdigest()[:16]


def parse_submissions(raw: dict[str, Any], allowed_forms: tuple[str, ...] = ("10-Q", "10-K"),
                      include_amendments: bool = False) -> pd.DataFrame:
    """Extract 10-Q/10-K (and optional /A) filings from a submissions.json payload.

    Schema follows RFC-01 §2.3.
    """
    cik = normalize_cik(raw.get("cik"))
    icid = internal_company_id(cik)

    recent = (raw.get("filings") or {}).get("recent") or {}
    cols = ["accessionNumber", "form", "filingDate", "reportDate", "acceptanceDateTime",
            "primaryDocument", "isXBRL", "isInlineXBRL"]
    arr = {c: recent.get(c) or [] for c in cols}
    n = min(len(v) for v in arr.values()) if arr["accessionNumber"] else 0

    rows = []
    for i in range(n):
        form = arr["form"][i]
        if form not in allowed_forms and not (include_amendments and form.endswith("/A")):
            continue
        accession = arr["accessionNumber"][i]
        rows.append({
            "filing_event_id": _filing_event_id(cik, accession),
            "internal_company_id": icid,
            "cik": cik,
            "accession_number": accession,
            "form_type": form,
            "filed_date": arr["filingDate"][i],
            "accepted_datetime": arr["acceptanceDateTime"][i],
            "report_period_end_date": arr["reportDate"][i],
            "primary_document": arr["primaryDocument"][i],
            "is_amendment": form.endswith("/A"),
            "source": "sec_submissions",
            "downloaded_at": datetime.now(tz=timezone.utc).isoformat(),
        })
    return pd.DataFrame(rows)
