"""Phase 34: cross-firm holdout test.

Splits the 1000-CIK universe into a "train pool" (70%) and "held-out test
companies" (30%). Trains LambdaRank tuned on the train pool only (across the
full 2003-2019 train+val time window). Predicts on the held-out companies
during the 2020-2026 test window. Backtests those predictions.

This is a much stronger out-of-sample test than the time-only split: the
model has NEVER seen any sample from these companies during fit, so any
generalization comes from the anonymous-feature representation, not from
company-specific memorization.

Deterministic split via hash of internal_company_id (no test snooping).
"""

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afp.features.encoder import AnonymousFeatureEncoder
from afp.features.price_features import (
    PriceFeatureConfig,
    compute_price_features,
    fit_scaler,
    transform_with_scaler,
)
from afp.models.datasets import build_tabular_dataset
from afp.models.train import attach_prediction_context


def _holdout_buckets(icid: str, n_buckets: int = 10) -> int:
    """Deterministic per-company bucket in [0, n_buckets)."""
    h = hashlib.sha1(icid.encode()).hexdigest()[:8]
    return int(h, 16) % n_buckets


def main(model_id: str = "lambdarank_company_holdout",
         holdout_fraction: float = 0.30,
         seed_bucket_start: int = 7):
    import lightgbm as lgb

    samples = pd.read_parquet("data/processed/event_samples.parquet")
    facts = pd.read_parquet("data/processed/financial_facts_long.parquet")
    prices = pd.read_parquet("data/processed/prices_daily.parquet")
    encoder = AnonymousFeatureEncoder.load("artifacts/feature_encoder/v1")

    # Per-company holdout bucket
    samples["_hbucket"] = samples["internal_company_id"].map(lambda x: _holdout_buckets(x))
    n_holdout = int(round(10 * holdout_fraction))           # 3 of 10 buckets
    holdout_buckets = set(range((seed_bucket_start) % 10,
                                (seed_bucket_start + n_holdout) % 10 or 10))
    if len(holdout_buckets) < n_holdout:
        # wrap-around
        holdout_buckets = set(((seed_bucket_start + i) % 10 for i in range(n_holdout)))
    is_holdout = samples["_hbucket"].isin(holdout_buckets)
    holdout_ciks = samples.loc[is_holdout, "internal_company_id"].unique()
    train_ciks = samples.loc[~is_holdout, "internal_company_id"].unique()
    print(f"holdout bucket(s) = {sorted(holdout_buckets)}: "
          f"{len(holdout_ciks)} held-out CIKs, {len(train_ciks)} train CIKs")

    # Standard time-split, but applied to companies in the train pool only
    pool = samples[~is_holdout].dropna(subset=["target_normalized_signal"]).copy()
    train = pool[pool["split"] == "train"].copy()
    val = pool[pool["split"] == "validation"].copy()
    # Test = held-out companies during test time window
    test = samples[is_holdout & (samples["split"] == "test")].dropna(
        subset=["target_normalized_signal"]).copy()
    print(f"train rows (70% CIKs, ≤2015-12-31):     {len(train)}")
    print(f"val rows   (70% CIKs, 2016-2019):       {len(val)}")
    print(f"test rows  (30% CIKs, ≥2020-01-01):     {len(test)}")

    feature_ids = encoder.artifact.feature_map.feature_ids
    pf_cfg = PriceFeatureConfig(enabled=True)
    pf_train, price_ids = compute_price_features(train, prices, pf_cfg)
    pf_val, _ = compute_price_features(val, prices, pf_cfg)
    pf_test, _ = compute_price_features(test, prices, pf_cfg)
    stats = fit_scaler(pf_train, price_ids)
    pf_train = transform_with_scaler(pf_train, price_ids, stats)
    pf_val = transform_with_scaler(pf_val, price_ids, stats)
    pf_test = transform_with_scaler(pf_test, price_ids, stats)

    def _ds(samples_df, price_df):
        scaled, missing, meta, sids = encoder.transform(samples_df, facts)
        targets = samples_df.set_index("sample_id")["target_normalized_signal"]
        base = build_tabular_dataset(scaled, missing, meta, sids, targets, feature_ids)
        sub = price_df.set_index("sample_id").reindex(base.sample_ids)
        extra = sub[price_ids].to_numpy(dtype=np.float64)
        extra_miss = sub[[f"{fid}_missing" for fid in price_ids]].to_numpy(dtype=np.float64)
        base.X = np.concatenate([base.X, extra, extra_miss], axis=1)
        return base, samples_df.set_index("sample_id").loc[base.sample_ids].reset_index()

    train_ds, train_rows = _ds(train, pf_train)
    val_ds, val_rows = _ds(val, pf_val)
    test_ds, test_rows = _ds(test, pf_test)

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

    train_data = lgb.Dataset(Xtr, label=label_tr, group=g_tr)
    val_data = lgb.Dataset(Xva, label=label_va, group=g_va, reference=train_data)
    params = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10, 50],
        "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 30,
        "feature_fraction": 0.3, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 1.0, "verbosity": -1,
        "max_position": 50, "label_gain": [pow(2, i) - 1 for i in range(31)],
    }
    print("training LambdaRank tuned on the 70%-CIK pool...")
    booster = lgb.train(params, train_data, num_boost_round=600,
                        valid_sets=[val_data],
                        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

    def _score_to_signal(scores, rows):
        df = rows.copy()
        df["score"] = scores
        df["q"] = pd.to_datetime(df["entry_date"]).dt.to_period("Q").astype(str)
        df["rk"] = df.groupby("q")["score"].rank(method="average", pct=True)
        return (2.0 * df["rk"].to_numpy() - 1.0)

    val_pred = _score_to_signal(booster.predict(Xva), val_rows)
    test_pred = _score_to_signal(booster.predict(Xte), test_rows)

    from afp.models.metrics import regression_metrics
    print("val metrics (70% CIKs, 2016-2019):", {k: round(v, 4) for k, v in regression_metrics(
        val_rows["target_normalized_signal"].to_numpy(), val_pred).items()})
    print("test metrics (30% HELD-OUT CIKs, ≥2020):", {k: round(v, 4) for k, v in regression_metrics(
        test_rows["target_normalized_signal"].to_numpy(), test_pred).items()})

    rows_out = []
    for sid, p in zip(sid_va, val_pred):
        rows_out.append({"sample_id": sid, "predicted_signal": float(p), "split": "validation"})
    for sid, p in zip(sid_te, test_pred):
        rows_out.append({"sample_id": sid, "predicted_signal": float(p), "split": "test"})
    preds = pd.DataFrame(rows_out)
    preds["model_id"] = model_id
    preds["model_type"] = "lambdarank_company_holdout"
    preds["prediction_created_at"] = datetime.now(tz=timezone.utc).isoformat()
    joined = attach_prediction_context(preds, samples)
    out = Path(f"artifacts/predictions/{model_id}.parquet")
    joined.to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
