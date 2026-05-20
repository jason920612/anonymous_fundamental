"""Phase 16: SIC sector extraction from SEC submissions JSON.

The SEC `submissions/CIK<cik>.json` payload includes `sicDescription` and
`sic` (the 4-digit SIC code). We map it down to the 1-digit SIC division so
the portfolio module has a coarse but actionable sector identifier.

This module produces only the metadata table; the portfolio module wires
sector caps separately. RFC-01 §2.5 / §3.3 allow sector data in the portfolio
module without leaking sector names to the prediction model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from afp.data.identifiers import internal_company_id, normalize_cik


# Coarse SIC division code → human-readable name (used only in reports).
SIC_DIVISIONS = {
    "0": "Agriculture, Forestry, Fishing",
    "1": "Mining/Construction",
    "2": "Manufacturing-1",
    "3": "Manufacturing-2",
    "4": "Transport/Utilities",
    "5": "Wholesale/Retail",
    "6": "Finance/Insurance/RealEstate",
    "7": "Services-1",
    "8": "Services-2/Health/Education",
    "9": "Public Admin",
}


def sic_to_division(sic: str | int | None) -> str | None:
    if sic is None:
        return None
    s = str(sic)
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(4)[0]


def parse_sector_from_submissions(payload: dict) -> dict | None:
    """Extract `{cik, internal_company_id, sic, sic_description, sic_division}`."""
    cik = payload.get("cik")
    if cik is None:
        return None
    cik10 = normalize_cik(cik)
    sic = payload.get("sic") or payload.get("sicCode")
    sic_desc = payload.get("sicDescription") or payload.get("sic_description")
    return {
        "cik": cik10,
        "internal_company_id": internal_company_id(cik10),
        "sic": str(sic) if sic else None,
        "sic_description": sic_desc,
        "sic_division": sic_to_division(sic),
    }


def build_sector_table_from_cache(submissions_dir: str | Path) -> pd.DataFrame:
    """Walk cached `data/raw/sec/submissions/CIK*.json` and produce a sectors DataFrame."""
    sub_dir = Path(submissions_dir)
    rows = []
    for path in sorted(sub_dir.glob("CIK*.json")):
        try:
            payload = json.loads(path.read_text())
            sector = parse_sector_from_submissions(payload)
            if sector is not None:
                rows.append(sector)
        except Exception:
            continue
    return pd.DataFrame(rows)
