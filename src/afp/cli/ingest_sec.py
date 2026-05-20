"""`afp ingest-sec` — download companies + submissions + companyfacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from afp.cli._common import log, resolve_config
from afp.data.ingest import run_sec_ingestion


def add_subparser(sub):
    p = sub.add_parser("ingest-sec", help="Download SEC companies + filings + facts")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--limit-ciks", type=int, default=None,
                   help="If set, only ingest the first N CIKs (useful for smoke runs).")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = run_sec_ingestion(cfg["data"], out, limit_ciks=args.limit_ciks)
    log.info("ingest_sec_done", extra=manifest)
    return 0
