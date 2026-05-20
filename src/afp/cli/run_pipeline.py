"""`afp run-pipeline` — orchestrate ingest → build-dataset → train → backtest."""

from __future__ import annotations

import argparse

from afp.cli import backtest as cmd_backtest
from afp.cli import build_dataset as cmd_build
from afp.cli import ingest_prices as cmd_prices
from afp.cli import ingest_sec as cmd_sec
from afp.cli import train as cmd_train
from afp.cli._common import log, resolve_config


def add_subparser(sub):
    p = sub.add_parser("run-pipeline",
                       help="End-to-end pipeline: ingest-sec → ingest-prices → build-dataset → train → backtest")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--skip-ingest", action="store_true",
                   help="Skip ingestion steps (assume parquet files already exist).")
    p.add_argument("--limit-ciks", type=int, default=50,
                   help="CIK limit during ingest-sec for a quick pipeline run.")
    p.add_argument("--model-kind", default=None)
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config)

    if not args.skip_ingest:
        sec_ns = argparse.Namespace(config=args.config, out_dir="data/processed",
                                    limit_ciks=args.limit_ciks)
        cmd_sec.run(sec_ns)
        prices_ns = argparse.Namespace(config=args.config,
                                       companies="data/processed/companies.parquet",
                                       out="data/processed/prices_daily.parquet",
                                       benchmarks_out="data/processed/benchmarks_daily.parquet",
                                       calendar_out="data/processed/trading_calendar.parquet",
                                       limit=args.limit_ciks)
        cmd_prices.run(prices_ns)

    build_ns = argparse.Namespace(config=args.config,
                                  filings="data/processed/filings.parquet",
                                  facts="data/processed/financial_facts_long.parquet",
                                  prices="data/processed/prices_daily.parquet",
                                  benchmarks="data/processed/benchmarks_daily.parquet",
                                  calendar="data/processed/trading_calendar.parquet",
                                  out_dir="data/processed",
                                  encoder_dir="artifacts/feature_encoder/v1")
    cmd_build.run(build_ns)

    train_ns = argparse.Namespace(config=args.config,
                                  samples="data/processed/event_samples.parquet",
                                  facts="data/processed/financial_facts_long.parquet",
                                  encoder_dir="artifacts/feature_encoder/v1",
                                  model_kind=args.model_kind,
                                  model_id=None,
                                  out_dir="artifacts/predictions")
    cmd_train.run(train_ns)

    kind = args.model_kind or cfg["model"]["type"]
    model_id = f"{kind}_v001"
    bt_ns = argparse.Namespace(config=args.config,
                               predictions=f"artifacts/predictions/{model_id}.parquet",
                               prices="data/processed/prices_daily.parquet",
                               benchmarks="data/processed/benchmarks_daily.parquet",
                               calendar="data/processed/trading_calendar.parquet",
                               out_dir="reports/backtest",
                               portfolio_id=None)
    cmd_backtest.run(bt_ns)

    log.info("pipeline_done")
    return 0
