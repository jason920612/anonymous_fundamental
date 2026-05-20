"""Ingest CIKs at positions 1000-1999 (after the original 1000) into a separate
processed/tier2 namespace, then build event_samples + transform with the
EXISTING encoder, predict with LambdaRank, and backtest.

This is the cross-universe robustness test: the model has never seen any of
these CIKs during training. If LambdaRank truly learned anonymous-feature
patterns rather than company memorization, performance should hold.
"""

import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afp.data.ingest import run_sec_ingestion
from afp.utils.config import load_config
from afp.utils.logging import get_logger

log = get_logger("tier2")


def main():
    cfg = load_config(ROOT / "configs" / "deployment_v2000.yaml")
    sec_cfg = cfg["data"]
    out_dir = Path("data/processed/tier2")
    out_dir.mkdir(parents=True, exist_ok=True)
    # raw will go to data/raw/sec/{submissions,companyfacts}/CIK* — shared cache.

    # Custom ingestion: pull CIKs 1000-1999 from the SEC ticker file
    from afp.data.parse_companies import parse_company_tickers
    from afp.data.sec_client import SecClient
    from afp.utils.io import write_parquet

    client = SecClient(
        user_agent=sec_cfg["sec"]["user_agent"],
        max_requests_per_second=sec_cfg["sec"].get("max_requests_per_second", 8),
    )
    print("downloading SEC company_tickers...")
    raw = client.company_tickers()
    all_companies = parse_company_tickers(raw)
    # Use positions 1000..1999
    tier2 = all_companies.iloc[1000:2000].reset_index(drop=True)
    print(f"tier2 universe: {len(tier2)} CIKs (positions 1000..1999)")
    write_parquet(tier2, out_dir / "companies.parquet")
    # Reuse the existing run_sec_ingestion but with a tier2-only list
    # Easiest: monkey-patch the cik list by writing a dummy companies parquet
    # and calling the full function. But the function reads from its own companies fetch.
    # Simpler: replicate inline.
    from afp.data.ingest import rebuild_sec_parquets_from_cache
    from afp.data.parse_companyfacts import parse_company_facts
    from afp.data.parse_submissions import parse_submissions
    from afp.data.sec_client import PermanentSecError

    raw_dir = Path("data/raw/sec")
    (raw_dir / "submissions").mkdir(parents=True, exist_ok=True)
    (raw_dir / "companyfacts").mkdir(parents=True, exist_ok=True)
    failed_marker = raw_dir / "companyfacts" / "_404_ciks.json"
    known_404: set[str] = set()
    if failed_marker.exists():
        try:
            known_404 = set(json.loads(failed_marker.read_text()))
        except Exception:
            known_404 = set()
    new_404: set[str] = set()

    allowed_forms = tuple(sec_cfg["forms"]["primary"])
    filings_frames = []
    facts_frames = []
    for i, cik in enumerate(tier2["cik"].tolist()):
        if cik in known_404:
            continue
        sub_cache = raw_dir / "submissions" / f"CIK{cik}.json"
        facts_cache = raw_dir / "companyfacts" / f"CIK{cik}.json"
        try:
            if sub_cache.exists():
                primary = json.loads(sub_cache.read_text())
                pages = [primary]
                for f in (primary.get("filings") or {}).get("files") or []:
                    name = f.get("name")
                    if not name:
                        continue
                    try:
                        page = client.get_json(f"https://data.sec.gov/submissions/{name}")
                        pages.append({"cik": cik, "filings": {"recent": page}})
                    except Exception:
                        continue
            else:
                pages = client.submissions_full(cik)
                sub_cache.write_text(json.dumps(pages[0]))

            for page in pages:
                df = parse_submissions(page, allowed_forms=allowed_forms)
                if not df.empty:
                    filings_frames.append(df)

            if facts_cache.exists():
                facts_payload = json.loads(facts_cache.read_text())
            else:
                try:
                    facts_payload = client.company_facts(cik)
                    facts_cache.write_text(json.dumps(facts_payload))
                except PermanentSecError as exc:
                    new_404.add(cik)
                    continue
            facts_frames.append(parse_company_facts(facts_payload))
        except Exception as exc:
            log.warning("cik_failed", extra={"cik": cik, "err": str(exc)})
        if (i + 1) % 50 == 0:
            log.info("tier2_progress", extra={"done": i + 1, "total": len(tier2),
                                              "new_404": len(new_404)})

    if new_404:
        failed_marker.write_text(json.dumps(sorted(known_404 | new_404)))

    filings = pd.concat(filings_frames, ignore_index=True) if filings_frames else pd.DataFrame()
    if not filings.empty:
        filings = filings.drop_duplicates(subset=["filing_event_id"]).reset_index(drop=True)
    facts = pd.concat(facts_frames, ignore_index=True) if facts_frames else pd.DataFrame()
    if not facts.empty:
        facts = facts.drop_duplicates(subset=["fact_id"]).reset_index(drop=True)
    write_parquet(filings, out_dir / "filings.parquet")
    write_parquet(facts, out_dir / "financial_facts_long.parquet")
    print(f"tier2: {len(filings)} filings, {len(facts)} facts written")


if __name__ == "__main__":
    main()
