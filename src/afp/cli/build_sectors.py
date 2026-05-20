"""`afp build-sectors` — extract SIC sector metadata from cached SEC submissions JSONs."""

from __future__ import annotations

import argparse
from pathlib import Path

from afp.cli._common import log, write_parquet
from afp.data.sectors import build_sector_table_from_cache


def add_subparser(sub):
    p = sub.add_parser("build-sectors", help="Extract SIC sector metadata from cached SEC submissions JSONs")
    p.add_argument("--submissions-dir", default="data/raw/sec/submissions")
    p.add_argument("--out", default="data/processed/sectors.parquet")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    table = build_sector_table_from_cache(args.submissions_dir)
    if table.empty:
        log.warning("build_sectors_empty")
        return 1
    write_parquet(table, args.out)
    log.info("build_sectors_done", extra={
        "n_companies": len(table),
        "n_sectors": int(table["sic_division"].nunique()),
        "out": args.out,
    })
    return 0
