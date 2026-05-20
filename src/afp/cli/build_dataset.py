"""`afp build-dataset` — assemble event samples + targets + dense feature inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from afp.cli._common import log, require_parquet, resolve_config, write_parquet
from afp.data.trading_calendar import TradingCalendar
from afp.features.encoder import AnonymousFeatureEncoder
from afp.features.sample_builder import build_event_samples
from afp.features.universe_filter import SampleUniverseConfig, apply_sample_universe_filter
from afp.targets.target_transform import TargetConfig, apply as apply_target
from afp.targets.volatility_scale import ScaleConfig, compute_ex_ante_scale_for_samples


def add_subparser(sub):
    p = sub.add_parser("build-dataset", help="Build event samples, targets, and feature inputs")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--filings", default="data/processed/filings.parquet")
    p.add_argument("--facts", default="data/processed/financial_facts_long.parquet")
    p.add_argument("--prices", default="data/processed/prices_daily.parquet")
    p.add_argument("--benchmarks", default="data/processed/benchmarks_daily.parquet")
    p.add_argument("--calendar", default="data/processed/trading_calendar.parquet")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--encoder-dir", default="artifacts/feature_encoder/v1")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config)
    filings = require_parquet(args.filings)
    facts = require_parquet(args.facts)
    prices = require_parquet(args.prices)
    benchmarks = require_parquet(args.benchmarks)
    cal_df = require_parquet(args.calendar)
    calendar = TradingCalendar.from_dates(cal_df["date"].tolist())

    splits = cfg["model"]["splits"]
    samples, excluded = build_event_samples(
        filings, prices, calendar, benchmarks,
        train_end_date=splits["train_end_date"],
        validation_end_date=splits["validation_end_date"],
    )
    log.info("samples_built", extra={"n_samples": len(samples), "n_excluded": len(excluded)})

    # Phase 12: optional point-in-time universe filter per RFC-01 §6.1
    sf_cfg_raw = (cfg.get("data") or {}).get("sample_filter", {}) or {}
    if sf_cfg_raw.get("enabled", False):
        sf_cfg = SampleUniverseConfig(**{k: v for k, v in sf_cfg_raw.items()
                                         if k in SampleUniverseConfig.__dataclass_fields__})
        sf_cfg.enabled = True
        samples, dropped_filter = apply_sample_universe_filter(samples, prices, sf_cfg)
        log.info("sample_filter_applied",
                 extra={"kept": len(samples), "dropped": len(dropped_filter)})

    scaled = compute_ex_ante_scale_for_samples(
        samples, prices, calendar,
        ScaleConfig(**{k: v for k, v in cfg["target"]["scale"].items()
                       if k in {"method", "daily_vol_lookback_days", "event_vol_lookback_events",
                                "min_event_history", "daily_weight", "event_weight",
                                "min_ex_ante_scale", "max_ex_ante_scale"}}),
    )

    # H2: inject sector column when the target mode needs it
    target_mode = cfg["target"].get("target_mode", "tanh_scaled")
    if target_mode == "sector_neutral_rank" and "sector" not in scaled.columns:
        sectors_path = Path("data/processed/sectors.parquet")
        if not sectors_path.exists():
            raise FileNotFoundError(
                "sector_neutral_rank target requires data/processed/sectors.parquet "
                "(run `afp build-sectors` first)")
        sec_df = require_parquet(sectors_path)
        scaled = scaled.merge(
            sec_df[["internal_company_id", "sic_division"]].rename(
                columns={"sic_division": "sector"}),
            on="internal_company_id", how="left",
        )

    samples_with_target = apply_target(
        scaled,
        TargetConfig(
            k=cfg["target"]["k"],
            target_mode=target_mode,
            rank_period=cfg["target"].get("rank_period", "Q"),
        ),
    )
    log.info("target_applied",
             extra={"mode": cfg["target"].get("target_mode", "tanh_scaled"),
                    "n_eligible": int(samples_with_target["target_normalized_signal"].notna().sum())})

    out = Path(args.out_dir)
    write_parquet(samples_with_target, out / "event_samples.parquet")
    write_parquet(excluded, out / "event_samples_excluded.parquet")

    train = samples_with_target[samples_with_target["split"] == "train"].dropna(
        subset=["target_normalized_signal"])
    derived_cfg = (cfg["features"].get("derived") or {})
    encoder = AnonymousFeatureEncoder(
        periods_back=cfg["features"]["periods_back"],
        min_company_count=cfg["features"]["min_company_count"],
        min_sample_coverage_pct=cfg["features"]["min_sample_coverage_pct"],
        max_missing_pct=cfg["features"]["max_missing_pct"],
        mapping_version=cfg["features"]["mapping_version"],
        derived_enabled=derived_cfg.get("enabled", False),
        derived_yoy_lookback=derived_cfg.get("yoy_lookback", 4),
    )
    encoder.fit(train, facts)
    encoder.save(args.encoder_dir)
    log.info("encoder_fit", extra={"num_features": encoder.artifact.num_features(),
                                   "n_train": len(train)})

    # Materialize dense inputs for downstream training step
    for split in ("train", "validation", "test"):
        rows = samples_with_target[samples_with_target["split"] == split].dropna(
            subset=["target_normalized_signal"])
        if rows.empty:
            continue
        scaled_arr, missing_arr, meta_arr, sids = encoder.transform(rows, facts)
        dense = encoder.to_dense_frame(scaled_arr, missing_arr, meta_arr, sids)
        write_parquet(dense, Path("data/model_inputs") / f"features_{split}.parquet")
    return 0
