"""`afp walk-forward` — quarterly retrain + stitched out-of-sample predictions (RFC-05 §9)."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from afp.cli._common import log, require_parquet, resolve_config, write_parquet
from afp.features.encoder import AnonymousFeatureEncoder
from afp.models.train import attach_prediction_context, train_and_predict


def add_subparser(sub):
    p = sub.add_parser("walk-forward",
                       help="Walk-forward retraining and stitched out-of-sample predictions")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--samples", default="data/processed/event_samples.parquet")
    p.add_argument("--facts", default="data/processed/financial_facts_long.parquet")
    p.add_argument("--encoder-dir", default="artifacts/feature_encoder/v1")
    p.add_argument("--model-kind", default=None)
    p.add_argument("--start", default=None, help="First out-of-sample date (default cfg validation end+1d)")
    p.add_argument("--end", default=None, help="Last out-of-sample date (default last test entry_date)")
    p.add_argument("--frequency", default="QS", help="Pandas offset alias for retrain frequency (QS=quarterly start, AS=annual start)")
    p.add_argument("--min-train-months", type=int, default=24,
                   help="Minimum months of training data required before first retrain.")
    p.add_argument("--out", default="artifacts/predictions/walk_forward.parquet")
    p.set_defaults(func=run)


def _window_bounds(samples: pd.DataFrame, start_date: date, end_date: date, freq: str):
    anchors = pd.date_range(start=start_date, end=end_date, freq=freq).date
    if len(anchors) == 0 or anchors[0] > start_date:
        anchors = [start_date] + list(anchors)
    bounds = []
    for i, a in enumerate(anchors):
        end = anchors[i + 1] if i + 1 < len(anchors) else end_date + pd.Timedelta(days=1).to_pytimedelta()
        bounds.append((a, end))
    return bounds


def run(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config)
    samples = require_parquet(args.samples)
    facts = require_parquet(args.facts)
    encoder = AnonymousFeatureEncoder.load(args.encoder_dir)

    samples = samples.dropna(subset=["target_normalized_signal", "entry_date"]).copy()
    samples["entry_date_ts"] = pd.to_datetime(samples["entry_date"])

    splits = cfg["model"]["splits"]
    start_date = pd.Timestamp(args.start or pd.Timestamp(splits["validation_end_date"])
                              + pd.Timedelta(days=1)).date()
    end_date = pd.Timestamp(args.end or samples["entry_date_ts"].max()).date()
    bounds = _window_bounds(samples, start_date, end_date, args.frequency)
    kind = args.model_kind or cfg["model"]["type"]
    model_cfg = cfg["model"].get(kind, {})

    log.info("walk_forward_start", extra={"n_windows": len(bounds), "model_kind": kind,
                                          "start": str(start_date), "end": str(end_date)})

    stitched_rows: list[pd.DataFrame] = []
    for i, (win_start, win_end) in enumerate(bounds):
        train_mask = samples["entry_date_ts"].dt.date < win_start
        if (samples.loc[train_mask, "entry_date_ts"].max() - samples["entry_date_ts"].min()
                < pd.Timedelta(days=args.min_train_months * 30)):
            log.info("window_skipped_insufficient_train", extra={"window": i,
                                                                 "start": str(win_start)})
            continue
        oos_mask = (samples["entry_date_ts"].dt.date >= win_start) & \
                   (samples["entry_date_ts"].dt.date < win_end)
        train = samples[train_mask]
        oos = samples[oos_mask]
        if oos.empty:
            continue

        # Reuse single encoder (already fit on initial train) — RFC-03 §13 forbids re-fitting scalers
        # on each window because that would leak validation-set statistics.
        empty = samples.iloc[0:0]
        res = train_and_predict(kind, encoder, train, oos, empty, facts,
                                model_config=model_cfg,
                                model_id=f"{kind}_walk_{i:03d}_{win_start}")
        rows = res.predictions[res.predictions["split"] == "validation"].copy()
        rows["window"] = i
        rows["window_start"] = str(win_start)
        rows["window_end"] = str(win_end)
        stitched_rows.append(rows)
        log.info("window_done", extra={"window": i, "start": str(win_start),
                                       "end": str(win_end), "n_train": len(train),
                                       "n_oos": len(oos), **res.validation_metrics})

    if not stitched_rows:
        log.warning("walk_forward_no_predictions")
        return 1

    stitched = pd.concat(stitched_rows, ignore_index=True)
    joined = attach_prediction_context(stitched, samples.drop(columns=["entry_date_ts"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(joined, out)
    log.info("walk_forward_done", extra={"n_predictions": len(joined), "out": str(out)})
    return 0
