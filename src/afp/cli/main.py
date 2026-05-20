"""`afp` — top-level CLI dispatch."""

from __future__ import annotations

import argparse
import sys

from afp.cli import backtest, build_dataset, ingest_prices, ingest_sec, run_pipeline, train
from afp.cli import build_sectors, diagnostics, ensemble, finalize, walk_forward


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="afp",
                                     description="Anonymous Fundamental Portfolio CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest_sec.add_subparser(sub)
    ingest_prices.add_subparser(sub)
    build_dataset.add_subparser(sub)
    train.add_subparser(sub)
    backtest.add_subparser(sub)
    run_pipeline.add_subparser(sub)
    walk_forward.add_subparser(sub)
    diagnostics.add_subparser(sub)
    finalize.add_subparser(sub)
    build_sectors.add_subparser(sub)
    ensemble.add_subparser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
