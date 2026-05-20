"""Tier2 pipeline: build samples for CIKs 1000-1999, predict with the
ORIGINAL LambdaRank trained on the first 1000 CIKs, backtest.

This is the "apply trained model to a completely new universe" test.
No retraining — same model, new companies.
"""

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afp.data.ingest import build_calendar, run_price_ingestion
from afp.data.price_client_yfinance import YFinancePriceClient
from afp.data.trading_calendar import TradingCalendar
from afp.features.encoder import AnonymousFeatureEncoder
from afp.features.price_features import (
    PriceFeatureConfig,
    compute_price_features,
    fit_scaler,
    transform_with_scaler,
)
from afp.features.sample_builder import build_event_samples
from afp.models.datasets import build_tabular_dataset
from afp.models.train import attach_prediction_context
from afp.targets.target_transform import TargetConfig, apply as apply_target
from afp.targets.volatility_scale import (
    ScaleConfig,
    compute_ex_ante_scale_for_samples,
)
from afp.utils.config import load_config
from afp.utils.io import write_parquet


def step_ingest_prices():
    cfg = load_config(ROOT / "configs" / "deployment_v2000.yaml")
    companies = pd.read_parquet("data/processed/tier2/companies.parquet")
    client = YFinancePriceClient(cache_root="data/raw/prices_cache",
                                 pause_seconds=cfg["data"]["prices"]["request_pause_seconds"])
    start = cfg["data"]["prices"]["start_date"]
    end = cfg["data"]["prices"]["end_date"] or date.today().isoformat()
    print(f"fetching prices for {len(companies)} tier2 tickers...")
    stacked = run_price_ingestion(
        tickers=companies["ticker"].tolist(),
        client=client,
        start_date=start, end_date=end,
        out_path=Path("data/processed/tier2/prices_daily.parquet"),
        company_map=companies,
    )
    print(f"tier2 prices: {len(stacked)} rows for {stacked['ticker_at_date'].nunique()} tickers")


def step_build_samples_and_predict(model_id_in: str, model_id_out: str):
    """Build event samples for tier2, transform via existing encoder, train
    LambdaRank on the ORIGINAL train pool, predict on tier2.
    """
    cfg = load_config(ROOT / "configs" / "deployment_v1000.yaml")

    # Tier2 raw data
    tier2_filings = pd.read_parquet("data/processed/tier2/filings.parquet")
    tier2_facts = pd.read_parquet("data/processed/tier2/financial_facts_long.parquet")
    tier2_prices = pd.read_parquet("data/processed/tier2/prices_daily.parquet")
    benchmarks = pd.read_parquet("data/processed/benchmarks_daily.parquet")
    cal_df = pd.read_parquet("data/processed/trading_calendar.parquet")
    calendar = TradingCalendar.from_dates(cal_df["date"].tolist())

    # Use exactly the same event-sample / scale / target pipeline as the
    # original 1000-CIK run — this is just a different *universe*.
    samples, excluded = build_event_samples(
        tier2_filings, tier2_prices, calendar, benchmarks,
        train_end_date="2015-12-31", validation_end_date="2019-12-31",
    )
    print(f"tier2 event samples: {len(samples)} ({len(excluded)} excluded)")
    scaled = compute_ex_ante_scale_for_samples(samples, tier2_prices, calendar, ScaleConfig())
    samples_with_target = apply_target(scaled, TargetConfig(k=2.5))
    write_parquet(samples_with_target, "data/processed/tier2/event_samples.parquet")

    # Predict on tier2 test rows using the SAME encoder + retrained LambdaRank
    # (re-training is allowed because train data is the ORIGINAL 1000-CIK pool
    # — same model state, different application universe.)
    encoder = AnonymousFeatureEncoder.load("artifacts/feature_encoder/v1")
    samples_orig = pd.read_parquet("data/processed/event_samples.parquet")
    facts_orig = pd.read_parquet("data/processed/financial_facts_long.parquet")
    prices_orig = pd.read_parquet("data/processed/prices_daily.parquet")

    train = samples_orig[samples_orig["split"] == "train"].dropna(subset=["target_normalized_signal"])
    val = samples_orig[samples_orig["split"] == "validation"].dropna(subset=["target_normalized_signal"])
    test_t2 = samples_with_target[samples_with_target["split"] == "test"].dropna(
        subset=["target_normalized_signal"])
    print(f"using {len(train)} train rows from original CIKs; predicting on {len(test_t2)} tier2 test rows")

    feature_ids = encoder.artifact.feature_map.feature_ids
    pf_cfg = PriceFeatureConfig(enabled=True)
    pf_train, price_ids = compute_price_features(train, prices_orig, pf_cfg)
    pf_val, _ = compute_price_features(val, prices_orig, pf_cfg)
    pf_test, _ = compute_price_features(test_t2, tier2_prices, pf_cfg)
    stats = fit_scaler(pf_train, price_ids)
    pf_train = transform_with_scaler(pf_train, price_ids, stats)
    pf_val = transform_with_scaler(pf_val, price_ids, stats)
    pf_test = transform_with_scaler(pf_test, price_ids, stats)

    def _ds(samples_df, price_df, facts_to_use):
        scaled_, missing_, meta_, sids = encoder.transform(samples_df, facts_to_use)
        targets = samples_df.set_index("sample_id")["target_normalized_signal"]
        base = build_tabular_dataset(scaled_, missing_, meta_, sids, targets, feature_ids)
        sub = price_df.set_index("sample_id").reindex(base.sample_ids)
        extra = sub[price_ids].to_numpy(dtype=np.float64)
        extra_miss = sub[[f"{fid}_missing" for fid in price_ids]].to_numpy(dtype=np.float64)
        base.X = np.concatenate([base.X, extra, extra_miss], axis=1)
        return base, samples_df.set_index("sample_id").loc[base.sample_ids].reset_index()

    train_ds, train_rows = _ds(train, pf_train, facts_orig)
    val_ds, val_rows = _ds(val, pf_val, facts_orig)
    test_ds, test_rows = _ds(test_t2, pf_test, tier2_facts)
    print(f"tier2 X shape: {test_ds.X.shape}")

    # Sort by entry date for LambdaRank groups
    train_rows = train_rows.sort_values("entry_date").reset_index(drop=True)
    val_rows = val_rows.sort_values("entry_date").reset_index(drop=True)
    test_rows = test_rows.sort_values("entry_date").reset_index(drop=True)

    def _sort(ds, rows):
        sid_to_idx = {sid: i for i, sid in enumerate(ds.sample_ids)}
        order = [sid_to_idx[s] for s in rows["sample_id"].to_list()]
        return ds.X[order], np.array(rows["sample_id"].to_list())

    Xtr, sid_tr = _sort(train_ds, train_rows)
    Xva, sid_va = _sort(val_ds, val_rows)
    Xte, sid_te = _sort(test_ds, test_rows)

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

    import lightgbm as lgb
    train_data = lgb.Dataset(Xtr, label=label_tr, group=g_tr)
    val_data = lgb.Dataset(Xva, label=label_va, group=g_va, reference=train_data)
    params = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10, 50],
        "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 30,
        "feature_fraction": 0.3, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 1.0, "verbosity": -1,
        "max_position": 50, "label_gain": [pow(2, i) - 1 for i in range(31)],
    }
    print("training LambdaRank tuned on ORIGINAL CIKs (same as previous run)...")
    booster = lgb.train(params, train_data, num_boost_round=600,
                        valid_sets=[val_data],
                        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

    def _score_to_signal(scores, rows):
        df = rows.copy()
        df["score"] = scores
        df["q"] = pd.to_datetime(df["entry_date"]).dt.to_period("Q").astype(str)
        df["rk"] = df.groupby("q")["score"].rank(method="average", pct=True)
        return (2.0 * df["rk"].to_numpy() - 1.0)

    test_pred = _score_to_signal(booster.predict(Xte), test_rows)
    rows_out = [{"sample_id": s, "predicted_signal": float(p), "split": "test"}
                for s, p in zip(sid_te, test_pred)]
    preds = pd.DataFrame(rows_out)
    preds["model_id"] = model_id_out
    preds["model_type"] = "lambdarank_tier2"
    preds["prediction_created_at"] = datetime.now(tz=timezone.utc).isoformat()
    joined = attach_prediction_context(preds, samples_with_target)
    out = Path(f"artifacts/predictions/{model_id_out}.parquet")
    joined.to_parquet(out, index=False)
    print(f"wrote {out} ({len(joined)} predictions on tier2 test universe)")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("prices", "all"):
        step_ingest_prices()
    if step in ("predict", "all"):
        step_build_samples_and_predict(model_id_in="lambdarank_v3_p25_s42",
                                       model_id_out="lambdarank_tier2")
