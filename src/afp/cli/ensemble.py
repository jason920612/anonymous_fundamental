"""`afp ensemble` — average predictions from multiple models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from afp.cli._common import log, require_parquet
from afp.models.ensemble import equal_weight_ensemble, rank_average_ensemble


def add_subparser(sub):
    p = sub.add_parser("ensemble", help="Equal-weight or rank-average ensemble of prediction parquets")
    p.add_argument("--predictions", nargs="+", required=True,
                   help="paths to two or more prediction parquets")
    p.add_argument("--method", choices=["mean", "rank"], default="mean")
    p.add_argument("--model-id", default="ensemble_v1")
    p.add_argument("--out", required=True)
    p.add_argument("--samples", default="data/processed/event_samples.parquet",
                   help="attach prediction context (entry_date, exit_date, ex_ante_scale)")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    if args.method == "rank":
        out_path = rank_average_ensemble(args.predictions, args.out, args.model_id)
    else:
        out_path = equal_weight_ensemble(args.predictions, args.out, args.model_id)
    log.info("ensemble_done", extra={"method": args.method, "n_inputs": len(args.predictions),
                                     "out": str(out_path)})
    # Attach context so the backtest CLI can read entry/exit/ex_ante_scale.
    samples = require_parquet(args.samples)
    ensemble = pd.read_parquet(out_path)
    keep = ["sample_id", "internal_company_id", "entry_date", "exit_date",
            "ex_ante_scale", "target_normalized_signal", "raw_log_return"]
    merged = ensemble.merge(samples[keep], on="sample_id", how="left")
    merged.to_parquet(out_path, index=False)
    return 0
