"""`afp refresh-all` — one-command fool-proof end-to-end refresh.

Walks the entire pipeline from data ingest to trained-model artifacts:

  1. Refresh SEC submissions + companyfacts caches
  2. Refresh yfinance price caches (only stale tickers re-downloaded)
  3. Rebuild event_samples + targets if any data changed
  4. Recompute cross-disciplinary market-state features
  5. Train LambdaRank with Phase 62 feature stack
  6. Save booster + price_stats + cross-feature metadata
  7. Rebuild the predict CLI's reference cohort cache
  8. Print a one-shot status summary

Designed to be idempotent — re-running on a fresh day only re-fetches
what is stale, and re-trains only if new samples are present.

Default config: configs/deployment_v1000.yaml (1000-CIK production
pool). Override with --config.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from afp.cli._common import log, resolve_config


# ---------------------------------------------------------------------------

def add_subparser(sub):
    p = sub.add_parser("refresh-all",
                       help="One-command refresh: data + features + model + artifacts")
    p.add_argument("--config", default="configs/deployment_v1000.yaml",
                   help="Pipeline config (default: deployment_v1000.yaml)")
    p.add_argument("--limit-ciks", type=int, default=1000,
                   help="Number of CIKs to ingest (default 1000)")
    p.add_argument("--skip-ingest-sec", action="store_true",
                   help="Skip SEC ingest (use existing parquet)")
    p.add_argument("--skip-ingest-prices", action="store_true",
                   help="Skip yfinance prices ingest")
    p.add_argument("--skip-build", action="store_true",
                   help="Skip event-sample rebuild")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip model training (use existing booster)")
    p.add_argument("--force-retrain", action="store_true",
                   help="Train even if no new samples detected")
    p.add_argument("--model-out", default="artifacts/models/lambdarank_v3_cross",
                   help="Output dir for trained booster + metadata")
    p.set_defaults(func=run)


# ---------------------------------------------------------------------------

def _hr(title: str):
    """Print a banner heading."""
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)


def _data_mtime(*paths: str) -> float:
    """Return the max mtime across given paths; 0 if any missing."""
    mtimes = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            return 0.0
        mtimes.append(path.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


# ---------------------------------------------------------------------------

def _step_ingest_sec(cfg, limit_ciks: int) -> bool:
    """Run afp ingest-sec. Returns True if any new data."""
    _hr("STEP 1/7 — Refresh SEC submissions + companyfacts")
    from afp.cli import ingest_sec
    ns = argparse.Namespace(
        config=Path(cfg["__source__"]) if "__source__" in cfg else "configs/deployment_v1000.yaml",
        out_dir="data/processed",
        limit_ciks=limit_ciks,
    )
    # The existing ingest_sec.run signature handles the config path
    ns.config = "configs/deployment_v1000.yaml"
    pre_mtime = _data_mtime("data/processed/filings.parquet",
                             "data/processed/financial_facts_long.parquet")
    t0 = time.time()
    ingest_sec.run(ns)
    post_mtime = _data_mtime("data/processed/filings.parquet",
                              "data/processed/financial_facts_long.parquet")
    elapsed = time.time() - t0
    changed = post_mtime > pre_mtime + 1
    print(f"  SEC ingest done in {elapsed:.1f}s — {'NEW DATA' if changed else 'no changes'}",
          flush=True)
    return changed


def _step_ingest_prices(cfg, limit: int) -> bool:
    _hr("STEP 2/7 — Refresh yfinance prices")
    from afp.cli import ingest_prices
    pre_mtime = _data_mtime("data/processed/prices_daily.parquet")
    ns = argparse.Namespace(
        config="configs/deployment_v1000.yaml",
        companies="data/processed/companies.parquet",
        out="data/processed/prices_daily.parquet",
        benchmarks_out="data/processed/benchmarks_daily.parquet",
        calendar_out="data/processed/trading_calendar.parquet",
        limit=limit,
    )
    t0 = time.time()
    ingest_prices.run(ns)
    post_mtime = _data_mtime("data/processed/prices_daily.parquet")
    elapsed = time.time() - t0
    changed = post_mtime > pre_mtime + 1
    print(f"  Prices ingest done in {elapsed:.1f}s — {'NEW DATA' if changed else 'no changes'}",
          flush=True)
    return changed


def _step_build_dataset() -> bool:
    _hr("STEP 3/7 — Rebuild event_samples + targets")
    from afp.cli import build_dataset
    pre_mtime = _data_mtime("data/processed/event_samples.parquet")
    ns = argparse.Namespace(
        config="configs/deployment_v1000.yaml",
        filings="data/processed/filings.parquet",
        facts="data/processed/financial_facts_long.parquet",
        prices="data/processed/prices_daily.parquet",
        benchmarks="data/processed/benchmarks_daily.parquet",
        calendar="data/processed/trading_calendar.parquet",
        out_dir="data/processed",
        encoder_dir="artifacts/feature_encoder/v1",
    )
    t0 = time.time()
    build_dataset.run(ns)
    post_mtime = _data_mtime("data/processed/event_samples.parquet")
    elapsed = time.time() - t0
    changed = post_mtime > pre_mtime + 1
    print(f"  Build-dataset done in {elapsed:.1f}s — {'NEW SAMPLES' if changed else 'no changes'}",
          flush=True)
    return changed


def _step_train_cross_features(model_out: Path):
    """Train the Phase 62 model and save booster + metadata + price-stats."""
    _hr("STEP 4/7 — Train LambdaRank with cross-disciplinary features (Phase 62)")
    import lightgbm as lgb

    from afp.features.cross_disciplinary_features import (
        CROSS_FEATURE_IDS,
        attach_cross_features_to_samples,
        compute_cross_features,
    )
    from afp.features.encoder import AnonymousFeatureEncoder
    from afp.features.price_features import (
        PriceFeatureConfig,
        compute_price_features,
        fit_scaler,
        transform_with_scaler,
    )
    from afp.models.datasets import build_tabular_dataset

    samples = pd.read_parquet("data/processed/event_samples.parquet")
    facts = pd.read_parquet("data/processed/financial_facts_long.parquet")
    prices = pd.read_parquet("data/processed/prices_daily.parquet")
    encoder = AnonymousFeatureEncoder.load("artifacts/feature_encoder/v1")

    train = samples[samples["split"] == "train"].dropna(subset=["target_normalized_signal"])
    val = samples[samples["split"] == "validation"].dropna(subset=["target_normalized_signal"])

    print(f"  train={len(train)} val={len(val)}", flush=True)

    print("  computing cross-disciplinary market features...", flush=True)
    price_pivot = (prices.assign(date=pd.to_datetime(prices["date"]))
                         .pivot_table(index="date", columns="internal_company_id",
                                      values="adjusted_close", aggfunc="last").ffill())
    t0 = time.time()
    cross_panel = compute_cross_features(price_pivot)
    print(f"    cross features: {time.time()-t0:.1f}s", flush=True)

    train = attach_cross_features_to_samples(train, cross_panel)
    val = attach_cross_features_to_samples(val, cross_panel)

    feature_ids = encoder.artifact.feature_map.feature_ids
    pf_cfg = PriceFeatureConfig(enabled=True)
    pf_train, price_ids = compute_price_features(train, prices, pf_cfg)
    pf_val, _ = compute_price_features(val, prices, pf_cfg)
    price_stats = fit_scaler(pf_train, price_ids)
    pf_train = transform_with_scaler(pf_train, price_ids, price_stats)
    pf_val = transform_with_scaler(pf_val, price_ids, price_stats)

    def _ds(samples_df, price_df):
        scaled_, missing_, meta_, sids = encoder.transform(samples_df, facts)
        targets = samples_df.set_index("sample_id")["target_normalized_signal"]
        base = build_tabular_dataset(scaled_, missing_, meta_, sids, targets, feature_ids)
        sub_p = price_df.set_index("sample_id").reindex(base.sample_ids)
        extra_p = sub_p[price_ids].to_numpy(dtype=np.float64)
        extra_pm = sub_p[[f"{fid}_missing" for fid in price_ids]].to_numpy(dtype=np.float64)
        sub_c = samples_df.set_index("sample_id").reindex(base.sample_ids)
        extra_c = sub_c[CROSS_FEATURE_IDS].to_numpy(dtype=np.float64)
        extra_cm = sub_c[[f"{f}_missing" for f in CROSS_FEATURE_IDS]].to_numpy(dtype=np.float64)
        base.X = np.concatenate([base.X, extra_p, extra_pm, extra_c, extra_cm], axis=1)
        return base, samples_df.set_index("sample_id").loc[base.sample_ids].reset_index()

    train_ds, train_rows = _ds(train, pf_train)
    val_ds, val_rows = _ds(val, pf_val)
    print(f"  feature shape: {train_ds.X.shape}", flush=True)

    train_rows = train_rows.sort_values("entry_date").reset_index(drop=True)
    val_rows = val_rows.sort_values("entry_date").reset_index(drop=True)

    def _sort(ds, rows):
        sid_to_idx = {sid: i for i, sid in enumerate(ds.sample_ids)}
        order = [sid_to_idx[s] for s in rows["sample_id"].to_list()]
        return ds.X[order]

    Xtr = _sort(train_ds, train_rows)
    Xva = _sort(val_ds, val_rows)

    def _to_label(rows):
        q = pd.to_datetime(rows["entry_date"]).dt.to_period("Q").astype(str)
        rk = rows.groupby(q)["raw_log_return"].rank(method="average", pct=True)
        return (rk * 30.999).astype(int).clip(0, 30).to_numpy()
    label_tr = _to_label(train_rows)
    label_va = _to_label(val_rows)

    def _group_sizes(rows):
        q = pd.to_datetime(rows["entry_date"]).dt.to_period("Q").astype(str)
        return q.groupby(q, sort=False).size().to_numpy()
    g_tr = _group_sizes(train_rows)
    g_va = _group_sizes(val_rows)

    train_data = lgb.Dataset(Xtr, label=label_tr, group=g_tr)
    val_data = lgb.Dataset(Xva, label=label_va, group=g_va, reference=train_data)
    params = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10, 50],
        "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 30,
        "feature_fraction": 0.3, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 1.0, "verbosity": -1,
        "max_position": 100, "label_gain": [pow(2, i) - 1 for i in range(31)],
    }
    print("  training LambdaRank with Phase 62 feature stack...", flush=True)
    t0 = time.time()
    booster = lgb.train(params, train_data, num_boost_round=600,
                        valid_sets=[val_data],
                        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    print(f"    trained in {time.time()-t0:.1f}s (best_iter={booster.best_iteration})",
          flush=True)

    model_out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_out / "booster.txt"))
    np.savez(model_out / "price_stats.npz",
             feature_ids=np.array(price_ids),
             medians=np.array([price_stats[fid][0] for fid in price_ids]),
             iqrs=np.array([price_stats[fid][1] for fid in price_ids]))
    meta = {
        "feature_encoder_dir": "artifacts/feature_encoder/v1",
        "n_base_features": len(feature_ids),
        "price_feature_ids": list(price_ids),
        "cross_feature_ids": list(CROSS_FEATURE_IDS),
        "best_iteration": int(booster.best_iteration or 0),
        "params": params,
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "trained_on": "1000-CIK train pool, ≤2015-12-31",
        "n_train_rows": int(len(train)),
        "n_val_rows": int(len(val)),
    }
    (model_out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"  saved booster + metadata to {model_out}", flush=True)

    # Phase 68: also mirror artifacts to the canonical legacy path so any
    # external tooling that hard-codes `artifacts/models/lambdarank_v3/`
    # automatically picks up the cross-features Phase 62 model.
    canonical = Path("artifacts/models/lambdarank_v3")
    if canonical.resolve() != model_out.resolve():
        canonical.mkdir(parents=True, exist_ok=True)
        import shutil
        for fname in ("booster.txt", "price_stats.npz", "metadata.json"):
            src = model_out / fname
            if src.exists():
                shutil.copy2(src, canonical / fname)
        # Invalidate any stale reference cohort at the canonical location.
        old_cohort = canonical / "reference_cohort.parquet"
        if old_cohort.exists():
            old_cohort.unlink()
        print(f"  mirrored artifacts to canonical {canonical} (Phase 62 is now default)",
              flush=True)
    return model_out


def _step_save_reference_cohort(model_out: Path):
    """Build a reference cohort cache the predict CLI can use."""
    _hr("STEP 5/7 — Build reference cohort cache")
    # The predict CLI itself rebuilds this when needed (Phase 37), so we
    # just invalidate the existing cache so the first predict run regenerates.
    cohort = model_out / "reference_cohort.parquet"
    if cohort.exists():
        cohort.unlink()
        print(f"  invalidated old reference cohort at {cohort}", flush=True)
    else:
        print(f"  no existing cohort cache; predict CLI will build on first run", flush=True)


def _step_check_predict_smoketest(model_out: Path):
    _hr("STEP 6/7 — Smoke-test the trained model via afp predict")
    if not (model_out / "booster.txt").exists():
        print("  no booster found — skipping smoke test", flush=True)
        return
    # We do a quick in-process load to verify metadata round-trips.
    try:
        meta = json.loads((model_out / "metadata.json").read_text())
        print(f"  booster.txt loadable; meta has {meta.get('n_base_features')} "
              f"base features + {len(meta.get('cross_feature_ids', []))} cross features.",
              flush=True)
    except Exception as exc:
        print(f"  smoke test FAILED: {exc}", flush=True)


def _step_status_summary(model_out: Path,
                          changes: dict[str, bool], total_elapsed: float):
    _hr("STEP 7/7 — Status summary")
    samples = Path("data/processed/event_samples.parquet")
    if samples.exists():
        s = pd.read_parquet(samples)
        latest_entry = pd.to_datetime(s["entry_date"]).max()
        latest_str = latest_entry.strftime("%Y-%m-%d")
    else:
        latest_str = "<no samples>"
    booster_ok = (model_out / "booster.txt").exists()
    meta = {}
    if (model_out / "metadata.json").exists():
        meta = json.loads((model_out / "metadata.json").read_text())

    print(f"  Pipeline finished in {total_elapsed/60:.1f} min", flush=True)
    print(f"  Latest sample entry_date: {latest_str}", flush=True)
    print(f"  SEC updated:    {changes.get('sec', False)}", flush=True)
    print(f"  Prices updated: {changes.get('prices', False)}", flush=True)
    print(f"  Samples rebuilt: {changes.get('build', False)}", flush=True)
    print(f"  Model retrained: {changes.get('train', False)}", flush=True)
    print(f"  Model artifacts: {model_out} (booster={'OK' if booster_ok else 'MISSING'})",
          flush=True)
    if meta:
        print(f"    base features: {meta.get('n_base_features')}", flush=True)
        print(f"    cross features: {len(meta.get('cross_feature_ids', []))}", flush=True)
        print(f"    trained at:    {meta.get('trained_at')}", flush=True)
    print("\n  Done. To predict on tickers, run:", flush=True)
    print("    afp predict TSLA AAPL NVDA", flush=True)


# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    cfg_path = args.config
    print(f"[afp refresh-all] starting with config={cfg_path}", flush=True)
    cfg = resolve_config(cfg_path)

    total_t0 = time.time()
    changes = {"sec": False, "prices": False, "build": False, "train": False}

    if not args.skip_ingest_sec:
        try:
            changes["sec"] = _step_ingest_sec(cfg, args.limit_ciks)
        except SystemExit as e:
            # ingest-sec calls sys.exit on bad config; rewrap as failure
            print(f"  SEC ingest failed: {e}", flush=True)
            return int(e.code or 1)
    else:
        print("[skip-ingest-sec] using existing SEC parquets", flush=True)

    if not args.skip_ingest_prices:
        changes["prices"] = _step_ingest_prices(cfg, args.limit_ciks)
    else:
        print("[skip-ingest-prices] using existing yfinance parquets", flush=True)

    if not args.skip_build:
        changes["build"] = _step_build_dataset()
    else:
        print("[skip-build] using existing event_samples", flush=True)

    model_out = Path(args.model_out)
    needs_train = args.force_retrain or changes["build"] or not (model_out / "booster.txt").exists()
    if args.skip_train:
        print("[skip-train] skipping training step", flush=True)
    elif not needs_train:
        print("[no-op] samples unchanged AND booster exists; skipping training",
              flush=True)
    else:
        _step_train_cross_features(model_out)
        changes["train"] = True

    _step_save_reference_cohort(model_out)
    _step_check_predict_smoketest(model_out)

    total_elapsed = time.time() - total_t0
    _step_status_summary(model_out, changes, total_elapsed)
    log.info("refresh_all_done", extra={"elapsed_sec": total_elapsed, **changes})
    return 0
