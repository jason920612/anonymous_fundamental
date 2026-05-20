"""`afp ingest-prices` — pull daily prices for all ingested tickers + benchmarks."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from afp.cli._common import log, require_parquet, resolve_config
from afp.data.ingest import build_calendar, run_price_ingestion
from afp.data.price_client import CsvPriceClient
from afp.data.price_client_yfinance import YFinancePriceClient


def add_subparser(sub):
    p = sub.add_parser("ingest-prices", help="Download daily prices for the ingested universe")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--companies", default="data/processed/companies.parquet")
    p.add_argument("--out", default="data/processed/prices_daily.parquet")
    p.add_argument("--benchmarks-out", default="data/processed/benchmarks_daily.parquet")
    p.add_argument("--calendar-out", default="data/processed/trading_calendar.parquet")
    p.add_argument("--limit", type=int, default=None,
                   help="Only fetch the first N tickers (useful for smoke runs).")
    p.set_defaults(func=run)


def _make_client(cfg: dict):
    src = cfg["data"]["prices"]["source"]
    if src == "yfinance":
        return YFinancePriceClient(cache_root=cfg["data"]["prices"]["cache_root"],
                                   pause_seconds=cfg["data"]["prices"]["request_pause_seconds"])
    if src == "csv":
        return CsvPriceClient(root=cfg["data"]["prices"]["csv_root"])
    raise ValueError(f"unknown price source: {src}")


def run(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config)
    client = _make_client(cfg)
    companies = require_parquet(args.companies)
    if args.limit:
        companies = companies.head(args.limit)
    start = cfg["data"]["prices"]["start_date"]
    end = cfg["data"]["prices"]["end_date"] or dt.date.today().isoformat()

    log.info("ingest_prices_start", extra={"n_tickers": len(companies)})
    stacked = run_price_ingestion(
        tickers=companies["ticker"].tolist(),
        client=client,
        start_date=start,
        end_date=end,
        out_path=Path(args.out),
        company_map=companies,
    )

    # Benchmarks
    benchmark_syms = cfg["data"]["benchmarks"]["symbols"]
    log.info("ingest_benchmarks", extra={"symbols": benchmark_syms})
    run_price_ingestion(
        tickers=benchmark_syms,
        client=client,
        start_date=start,
        end_date=end,
        out_path=Path(args.benchmarks_out),
    )

    # Trading calendar — span the union of price dates with a small buffer
    if not stacked.empty:
        start_cal = min(stacked["date"].min(), __import__("pandas").Timestamp(start).date())
        end_cal = max(stacked["date"].max(), __import__("pandas").Timestamp(end).date())
        build_calendar(str(start_cal), str(end_cal), Path(args.calendar_out))
        log.info("calendar_built", extra={"start": str(start_cal), "end": str(end_cal)})

    return 0
