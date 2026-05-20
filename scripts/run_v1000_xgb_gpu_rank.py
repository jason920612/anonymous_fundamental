"""H3 (GPU): XGBoost CUDA with rank:pairwise objective.

XGBoost ships a GPU-accelerated learning-to-rank objective which gives us
LambdaRank-style training entirely on the RTX 4060 — no LightGBM CPU dependency.
"""

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
from afp.models.metrics import regression_metrics
from afp.models.train import attach_prediction_context


def main(model_id: str = "xgb_gpu_rank"):
    import xgboost as xgb
    print("xgboost device check:", xgb.__version__, "CUDA via tree_method=hist + device=cuda")

    samples = pd.read_parquet("data/processed/event_samples.parquet")
    facts = pd.read_parquet("data/processed/financial_facts_long.parquet")
    prices = pd.read_parquet("data/processed/prices_daily.parquet")
    encoder = AnonymousFeatureEncoder.load("artifacts/feature_encoder/v1")

    train = samples[samples["split"] == "train"].dropna(subset=["target_normalized_signal"])
    val = samples[samples["split"] == "validation"].dropna(subset=["target_normalized_signal"])
    test = samples[samples["split"] == "test"].dropna(subset=["target_normalized_signal"])

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
        return base

    train_ds = _ds(train, pf_train)
    val_ds = _ds(val, pf_val)
    test_ds = _ds(test, pf_test)
    print(f"X shape: {train_ds.X.shape}")

    # Build per-quarter group indices for ranking
    def _qid_array(samples_df, ds_sample_ids):
        idx = samples_df.set_index("sample_id").loc[ds_sample_ids]
        return pd.to_datetime(idx["entry_date"]).dt.to_period("Q").astype(str).to_numpy()

    qid_tr = _qid_array(train, train_ds.sample_ids)
    qid_va = _qid_array(val, val_ds.sample_ids)

    # Sort each set so groups are contiguous (required by XGB rank API)
    def _sort_by_group(X, y, sids, qids):
        order = np.argsort(qids, kind="stable")
        return X[order], y[order], np.array(sids)[order], qids[order]

    Xtr, ytr, sid_tr, q_tr = _sort_by_group(train_ds.X, train_ds.y, train_ds.sample_ids, qid_tr)
    Xva, yva, sid_va, q_va = _sort_by_group(val_ds.X, val_ds.y, val_ds.sample_ids, qid_va)

    # Per-quarter pct_rank as label (higher = better realized return)
    def _label(samples_df, sids, qids):
        idx = samples_df.set_index("sample_id").loc[sids].reset_index()
        idx["_q"] = qids
        idx["_rk"] = idx.groupby("_q")["raw_log_return"].rank(method="average", pct=True)
        return (idx["_rk"] * 31.999).astype(int).clip(0, 31).to_numpy()

    label_tr = _label(train, sid_tr, q_tr)
    label_va = _label(val, sid_va, q_va)

    g_tr = pd.Series(q_tr).value_counts(sort=False).reindex(pd.unique(q_tr)).to_numpy()
    g_va = pd.Series(q_va).value_counts(sort=False).reindex(pd.unique(q_va)).to_numpy()

    dtrain = xgb.DMatrix(Xtr, label=label_tr)
    dtrain.set_group(g_tr)
    dval = xgb.DMatrix(Xva, label=label_va)
    dval.set_group(g_va)
    dtest = xgb.DMatrix(test_ds.X)

    params = {
        "objective": "rank:pairwise",
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.5,
        "eval_metric": "ndcg@50",
        "verbosity": 0,
    }
    booster = xgb.train(
        params, dtrain,
        num_boost_round=300,
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=0,
    )

    def _score_to_signal(scores, samples_df, sids):
        df = samples_df.set_index("sample_id").loc[sids].reset_index()
        df["score"] = scores
        df["q"] = pd.to_datetime(df["entry_date"]).dt.to_period("Q").astype(str)
        df["rk"] = df.groupby("q")["score"].rank(method="average", pct=True)
        return (2.0 * df["rk"].to_numpy() - 1.0)

    val_pred = _score_to_signal(booster.predict(dval), val, sid_va)
    test_pred = _score_to_signal(booster.predict(dtest), test, test_ds.sample_ids)
    print("val metrics:", {k: round(v, 4) for k, v in regression_metrics(yva, val_pred).items()})

    rows_out = []
    for sid, p in zip(sid_va, val_pred):
        rows_out.append({"sample_id": sid, "predicted_signal": float(p), "split": "validation"})
    for sid, p in zip(test_ds.sample_ids, test_pred):
        rows_out.append({"sample_id": sid, "predicted_signal": float(p), "split": "test"})
    preds = pd.DataFrame(rows_out)
    preds["model_id"] = model_id
    preds["model_type"] = "xgb_gpu_rank"
    preds["prediction_created_at"] = datetime.now(tz=timezone.utc).isoformat()
    joined = attach_prediction_context(preds, samples)
    out = Path(f"artifacts/predictions/{model_id}.parquet")
    joined.to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
