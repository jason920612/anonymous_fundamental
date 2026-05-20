"""`afp train` — fit a model and emit predictions parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

from afp.cli._common import log, require_parquet, resolve_config, write_parquet
from afp.features.encoder import AnonymousFeatureEncoder
from afp.models.train import attach_prediction_context, train_and_predict


def add_subparser(sub):
    p = sub.add_parser("train", help="Train a model and emit predictions")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--samples", default="data/processed/event_samples.parquet")
    p.add_argument("--facts", default="data/processed/financial_facts_long.parquet")
    p.add_argument("--encoder-dir", default="artifacts/feature_encoder/v1")
    p.add_argument("--model-kind", default=None,
                   help="Override config model.type (e.g. ridge, gbt, lightgbm, constant_zero)")
    p.add_argument("--model-id", default=None)
    p.add_argument("--out-dir", default="artifacts/predictions")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config)
    samples = require_parquet(args.samples)
    facts = require_parquet(args.facts)
    enc = AnonymousFeatureEncoder.load(args.encoder_dir)

    train = samples[samples["split"] == "train"].dropna(subset=["target_normalized_signal"])
    val = samples[samples["split"] == "validation"].dropna(subset=["target_normalized_signal"])
    test = samples[samples["split"] == "test"].dropna(subset=["target_normalized_signal"])

    kind = args.model_kind or cfg["model"]["type"]
    model_cfg = cfg["model"].get(kind, {})
    log.info("train_start", extra={"model_kind": kind, "n_train": len(train)})

    sw_cfg = (cfg["model"].get("sample_weighting") or {})
    res = train_and_predict(kind, enc, train, val, test, facts,
                            model_config=model_cfg, model_id=args.model_id,
                            weight_by_company=sw_cfg.get("by_company", False),
                            weight_by_quarter=sw_cfg.get("by_quarter", False))
    log.info("train_done", extra={"model_id": res.model_id, **res.validation_metrics})

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joined = attach_prediction_context(res.predictions, samples)
    write_parquet(joined, out / f"{res.model_id}.parquet")
    return 0
