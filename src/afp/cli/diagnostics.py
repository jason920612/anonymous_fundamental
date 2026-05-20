"""`afp diagnostics` — signal-level diagnostics for a predictions parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from afp.cli._common import log, require_parquet
from afp.models.diagnostics import DiagnosticsConfig, build_diagnostic_report


def add_subparser(sub):
    p = sub.add_parser("diagnostics", help="Signal-level diagnostics (decile table, "
                                           "Spearman by quarter, top-bottom spread, random comparison)")
    p.add_argument("--predictions", required=True,
                   help="predictions parquet with predicted_signal + raw_log_return + entry_date")
    p.add_argument("--out-dir", default="reports/diagnostics")
    p.add_argument("--portfolio-id", default=None)
    p.add_argument("--split", default="test",
                   help="Filter to a split column value (test|validation|train). Use '' for all.")
    p.add_argument("--n-buckets", type=int, default=10)
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    predictions = require_parquet(args.predictions)
    if "predicted_signal" not in predictions.columns:
        raise ValueError("predictions parquet missing predicted_signal column")
    if "raw_log_return" not in predictions.columns:
        raise ValueError("predictions parquet missing raw_log_return (run `afp train` first)")

    split = args.split if args.split else None
    cfg = DiagnosticsConfig(n_buckets=args.n_buckets)
    report = build_diagnostic_report(predictions, cfg, split=split)

    portfolio_id = args.portfolio_id or Path(args.predictions).stem
    out_dir = Path(args.out_dir) / portfolio_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("diagnostics_done", extra={
        "portfolio_id": portfolio_id,
        "n_samples": report["n_samples"],
        "top_minus_bottom_log_return": report["top_minus_bottom_log_return"],
        **report["positive_signal_vs_random"],
    })
    return 0
