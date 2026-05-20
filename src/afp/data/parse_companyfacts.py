"""Flatten SEC `companyfacts.json` into long-format financial facts."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from afp.data.identifiers import internal_company_id, normalize_cik


def _fact_id(cik: str, taxonomy: str, concept: str, accession: str, period_end: str, unit: str) -> str:
    key = "|".join([cik, taxonomy, concept, accession, period_end or "", unit])
    return "FACT_" + hashlib.sha1(key.encode()).hexdigest()[:20]


def parse_company_facts(raw: dict[str, Any]) -> pd.DataFrame:
    """Convert a companyfacts payload to a long-format DataFrame.

    Schema follows RFC-01 §2.2 and RFC-02 §7.3. Raw concept names are preserved
    here; anonymization happens in Phase 5.
    """
    cik = normalize_cik(raw.get("cik"))
    icid = internal_company_id(cik)
    facts = raw.get("facts") or {}

    rows = []
    downloaded_at = datetime.now(tz=timezone.utc).isoformat()
    for taxonomy, concepts in facts.items():
        for concept, payload in concepts.items():
            units = (payload or {}).get("units") or {}
            for unit, observations in units.items():
                for obs in observations or []:
                    accession = obs.get("accn") or ""
                    period_end = obs.get("end")
                    form = obs.get("form")
                    raw_val = obs.get("val")
                    # SEC reports some share counts / currencies in values larger than
                    # int64; pandas/pyarrow rejects mixed object columns containing them.
                    # Coerce everything to float64.
                    try:
                        value = float(raw_val) if raw_val is not None else None
                    except (TypeError, ValueError):
                        value = None
                    rows.append({
                        "fact_id": _fact_id(cik, taxonomy, concept, accession, period_end, unit),
                        "internal_company_id": icid,
                        "cik": cik,
                        "taxonomy": taxonomy,
                        "concept_name": concept,
                        "unit": unit,
                        "value": value,
                        "period_start_date": obs.get("start"),
                        "period_end_date": period_end,
                        "fiscal_year": obs.get("fy"),
                        "fiscal_period": obs.get("fp"),
                        "form_type": form,
                        "accession_number": accession,
                        "accepted_datetime": obs.get("filed"),
                        "filed_date": obs.get("filed"),
                        "frame": obs.get("frame"),
                        "source": "sec_companyfacts",
                        "downloaded_at": downloaded_at,
                    })
    return pd.DataFrame(rows)
