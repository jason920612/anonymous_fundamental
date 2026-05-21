"""Ingest CIKs at positions 2000-2999 — second cross-universe holdout (tier3).

Same logic as scripts/ingest_tier2.py but pulls a different slice of the SEC
company ticker file. Output namespace: data/processed/tier3/.

The model was trained on positions 0-999 (tier1) and tested in Phase 34 on
positions 1000-1999 (tier2). This tier3 = positions 2000-2999 — a THIRD
disjoint universe, never seen by the model, and never used to evaluate any
allocator/wrapper choice.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afp.utils.config import load_config
from afp.utils.logging import get_logger

log = get_logger("tier3")


def main():
    cfg = load_config(ROOT / "configs" / "deployment_v3000.yaml")
    sec_cfg = cfg["data"]
    out_dir = Path("data/processed/tier3")
    out_dir.mkdir(parents=True, exist_ok=True)

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
    tier3 = all_companies.iloc[2000:3000].reset_index(drop=True)
    print(f"tier3 universe: {len(tier3)} CIKs (positions 2000..2999)")
    write_parquet(tier3, out_dir / "companies.parquet")

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
    for i, cik in enumerate(tier3["cik"].tolist()):
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
                except PermanentSecError:
                    new_404.add(cik)
                    continue
            facts_frames.append(parse_company_facts(facts_payload))
        except Exception as exc:
            log.warning("cik_failed", extra={"cik": cik, "err": str(exc)})
        if (i + 1) % 50 == 0:
            log.info("tier3_progress", extra={"done": i + 1, "total": len(tier3),
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
    print(f"tier3: {len(filings)} filings, {len(facts)} facts written")


if __name__ == "__main__":
    main()
