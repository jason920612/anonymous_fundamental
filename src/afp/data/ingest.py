"""High-level ingestion orchestration.

Network-aware paths (`run_sec_ingestion`) require live access to data.sec.gov;
offline tests use the parse_* functions directly.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from afp.data.parse_companies import parse_company_tickers
from afp.data.parse_companyfacts import parse_company_facts
from afp.data.parse_submissions import parse_submissions
from afp.data.price_client import PriceClient, stack_price_frames, validate_price_frame
from afp.data.sec_client import PermanentSecError, SecClient
from afp.data.trading_calendar import TradingCalendar
from afp.utils.io import write_parquet
from afp.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------

def run_sec_ingestion(
    cfg_data: dict,
    out_dir: Path,
    limit_ciks: int | None = None,
) -> dict:
    """End-to-end SEC ingestion: companies → submissions → companyfacts.

    Writes:
      out_dir/companies.parquet
      out_dir/filings.parquet
      out_dir/financial_facts_long.parquet
      out_dir/manifest.json
    """
    sec_cfg = cfg_data["sec"]
    client = SecClient(
        user_agent=sec_cfg["user_agent"],
        max_requests_per_second=sec_cfg.get("max_requests_per_second", 8),
        retry_attempts=sec_cfg["retry"]["max_attempts"],
        retry_backoff_seconds=tuple(sec_cfg["retry"]["backoff_seconds"]),
    )
    out_dir = Path(out_dir)
    raw_dir = out_dir.parent / "raw" / "sec"

    log.info("ingest_companies_start")
    tickers_raw = client.company_tickers()
    (raw_dir / "company_tickers").mkdir(parents=True, exist_ok=True)
    (raw_dir / "company_tickers" / "company_tickers.json").write_text(json.dumps(tickers_raw))
    companies = parse_company_tickers(tickers_raw)
    write_parquet(companies, out_dir / "companies.parquet")
    log.info("ingest_companies_done", extra={"n_companies": len(companies)})

    ciks = companies["cik"].tolist()
    if limit_ciks:
        ciks = ciks[:limit_ciks]

    filings_frames: list[pd.DataFrame] = []
    facts_frames: list[pd.DataFrame] = []
    allowed_forms = tuple(cfg_data["forms"]["primary"])
    include_amendments = cfg_data["forms"].get("include_amendments_as_events", False)
    (raw_dir / "submissions").mkdir(parents=True, exist_ok=True)
    (raw_dir / "companyfacts").mkdir(parents=True, exist_ok=True)

    # Resumable: remember CIKs known to 404 so reruns skip them.
    failed_marker = raw_dir / "companyfacts" / "_404_ciks.json"
    known_404: set[str] = set()
    if failed_marker.exists():
        try:
            known_404 = set(json.loads(failed_marker.read_text()))
        except Exception:
            known_404 = set()
    new_404: set[str] = set()

    for i, cik in enumerate(ciks):
        if cik in known_404:
            continue
        try:
            sub_cache = raw_dir / "submissions" / f"CIK{cik}.json"
            facts_cache = raw_dir / "companyfacts" / f"CIK{cik}.json"

            # Submissions: reuse cache if present, otherwise fetch (incl. pagination).
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
                df = parse_submissions(page, allowed_forms=allowed_forms,
                                       include_amendments=include_amendments)
                if not df.empty:
                    filings_frames.append(df)

            # Company facts: cached or fetched. Permanent 404 → mark and skip.
            if facts_cache.exists():
                facts_payload = json.loads(facts_cache.read_text())
            else:
                try:
                    facts_payload = client.company_facts(cik)
                    facts_cache.write_text(json.dumps(facts_payload))
                except PermanentSecError as exc:
                    log.warning("cik_no_facts", extra={"cik": cik, "status": exc.status_code})
                    new_404.add(cik)
                    continue
            facts_frames.append(parse_company_facts(facts_payload))
        except Exception as exc:
            log.warning("cik_failed", extra={"cik": cik, "err": str(exc)})
        if (i + 1) % 50 == 0:
            log.info("ingest_progress", extra={"done": i + 1, "total": len(ciks),
                                               "new_404": len(new_404),
                                               "known_404": len(known_404)})

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

    manifest = {
        "companies": int(len(companies)),
        "filings": int(len(filings)),
        "facts": int(len(facts)),
        "ciks_processed": int(len(ciks)),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("ingest_complete", extra=manifest)
    return manifest


def run_price_ingestion(
    tickers: Iterable[str],
    client: PriceClient,
    start_date: str | date,
    end_date: str | date,
    out_path: Path,
    company_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fetch + stack daily prices for the given tickers. RFC-01 §3.2."""
    frames: list[tuple[str, pd.DataFrame]] = []
    warnings: list[str] = []
    for t in tickers:
        try:
            df = client.get_daily_prices(t, start_date, end_date)
            warnings.extend(validate_price_frame(df, t))
            frames.append((t, df))
        except Exception as exc:
            log.warning("price_fetch_failed", extra={"ticker": t, "err": str(exc)})

    stacked = stack_price_frames(frames)
    if company_map is not None and not stacked.empty:
        mapping = company_map[["internal_company_id", "ticker"]].rename(columns={"ticker": "ticker_at_date"})
        stacked = stacked.merge(mapping, on="ticker_at_date", how="left")

    write_parquet(stacked, out_path)
    if warnings:
        log.warning("price_quality_warnings", extra={"count": len(warnings)})
    return stacked


def build_calendar(start: str, end: str, out_path: Path) -> TradingCalendar:
    cal = TradingCalendar.build(start, end)
    write_parquet(cal.to_frame(), out_path)
    return cal


# ---------------------------------------------------------------------------

def rebuild_sec_parquets_from_cache(out_dir: Path,
                                    allowed_forms: tuple[str, ...] = ("10-Q", "10-K"),
                                    include_amendments: bool = False) -> dict:
    """Re-parse cached SEC raw JSONs into parquet. Pure-offline, no network.

    Useful when the in-memory accumulation finished but the final write failed
    (e.g. a pandas/pyarrow type error). Walks `out_dir/../raw/sec/{submissions,
    companyfacts}/CIK*.json` and rebuilds `filings.parquet` +
    `financial_facts_long.parquet`. Companies parquet must already exist.
    """
    out_dir = Path(out_dir)
    raw_dir = out_dir.parent / "raw" / "sec"
    sub_dir = raw_dir / "submissions"
    facts_dir = raw_dir / "companyfacts"

    log.info("rebuild_from_cache_start", extra={
        "submissions": len(list(sub_dir.glob("CIK*.json"))),
        "companyfacts": len(list(facts_dir.glob("CIK*.json"))),
    })

    filings_frames: list[pd.DataFrame] = []
    facts_frames: list[pd.DataFrame] = []
    for path in sorted(sub_dir.glob("CIK*.json")):
        try:
            primary = json.loads(path.read_text())
            df = parse_submissions(primary, allowed_forms=allowed_forms,
                                   include_amendments=include_amendments)
            if not df.empty:
                filings_frames.append(df)
        except Exception as exc:
            log.warning("rebuild_submissions_failed",
                        extra={"path": str(path), "err": str(exc)})

    for path in sorted(facts_dir.glob("CIK*.json")):
        try:
            payload = json.loads(path.read_text())
            facts_frames.append(parse_company_facts(payload))
        except Exception as exc:
            log.warning("rebuild_facts_failed",
                        extra={"path": str(path), "err": str(exc)})

    filings = pd.concat(filings_frames, ignore_index=True) if filings_frames else pd.DataFrame()
    if not filings.empty:
        filings = filings.drop_duplicates(subset=["filing_event_id"]).reset_index(drop=True)
    facts = pd.concat(facts_frames, ignore_index=True) if facts_frames else pd.DataFrame()
    if not facts.empty:
        facts = facts.drop_duplicates(subset=["fact_id"]).reset_index(drop=True)
    write_parquet(filings, out_dir / "filings.parquet")
    write_parquet(facts, out_dir / "financial_facts_long.parquet")

    manifest = {"filings": int(len(filings)), "facts": int(len(facts))}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("rebuild_from_cache_done", extra=manifest)
    return manifest
