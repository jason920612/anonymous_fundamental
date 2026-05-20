"""H3: LightGBM LambdaRank with quarter as the group key.

LambdaRank optimizes pairwise ranking within each group — exactly the
quarter cohort structure we want for cross-sectional stock selection.
Label is pct_rank within the same quarter (higher = better future
ranking).
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


def main(model_id: str = "lgbm_lambdarank"):
    import lightgbm as lgb

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
        return base, samples_df.set_index("sample_id").loc[base.sample_ids].reset_index()

    train_ds, train_rows = _ds(train, pf_train)
    val_ds, val_rows = _ds(val, pf_val)
    test_ds, test_rows = _ds(test, pf_test)

    # Group by entry quarter
    def _group_sizes(rows):
        q = pd.to_datetime(rows["entry_date"]).dt.to_period("Q").astype(str)
        # rows already ordered as in dataset — group counts in order of appearance
        return q.groupby(q, sort=False).size().to_numpy()

    train_rows = train_rows.sort_values("entry_date").reset_index(drop=True)
    val_rows = val_rows.sort_values("entry_date").reset_index(drop=True)
    test_rows = test_rows.sort_values("entry_date").reset_index(drop=True)

    # Re-align X to sorted order (LambdaRank wants groups contiguous)
    def _sort_ds(ds, rows):
        sid_to_idx = {sid: i for i, sid in enumerate(ds.sample_ids)}
        order = [sid_to_idx[s] for s in rows["sample_id"].to_list()]
        return ds.X[order], ds.y[order], np.array(rows["sample_id"].to_list())

    Xtr, ytr, sid_tr = _sort_ds(train_ds, train_rows)
    Xva, yva, sid_va = _sort_ds(val_ds, val_rows)
    Xte, yte, sid_te = _sort_ds(test_ds, test_rows)

    # LambdaRank needs integer-positive labels (higher = better). Bucket pct_rank to ints.
    def _to_label(samples_df):
        q = pd.to_datetime(samples_df["entry_date"]).dt.to_period("Q").astype(str)
        rk = samples_df.groupby(q)["raw_log_return"].rank(method="average", pct=True)
        return (rk * 30.999).astype(int).clip(0, 30).to_numpy()

    label_tr = _to_label(train_rows)
    label_va = _to_label(val_rows)

    g_tr = _group_sizes(train_rows)
    g_va = _group_sizes(val_rows)
    print(f"train rows {len(ytr)} in {len(g_tr)} groups; val {len(yva)} in {len(g_va)} groups")

    train_data = lgb.Dataset(Xtr, label=label_tr, group=g_tr)
    val_data = lgb.Dataset(Xva, label=label_va, group=g_va, reference=train_data)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 50],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.5,
        "verbosity": -1,
    }
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    # LambdaRank outputs raw scores; map to [-1,1] via cohort pct_rank then 2x-1
    def _score_to_signal(scores, rows):
        df = rows.copy()
        df["score"] = scores
        df["q"] = pd.to_datetime(df["entry_date"]).dt.to_period("Q").astype(str)
        df["rk"] = df.groupby("q")["score"].rank(method="average", pct=True)
        return (2.0 * df["rk"].to_numpy() - 1.0)

    val_pred = _score_to_signal(booster.predict(Xva), val_rows)
    test_pred = _score_to_signal(booster.predict(Xte), test_rows)
    print("val metrics:", {k: round(v, 4) for k, v in regression_metrics(yva, val_pred).items()})

    rows_out = []
    for sid, p in zip(sid_va, val_pred):
        rows_out.append({"sample_id": sid, "predicted_signal": float(p), "split": "validation"})
    for sid, p in zip(sid_te, test_pred):
        rows_out.append({"sample_id": sid, "predicted_signal": float(p), "split": "test"})
    preds = pd.DataFrame(rows_out)
    preds["model_id"] = model_id
    preds["model_type"] = "lambdarank"
    preds["prediction_created_at"] = datetime.now(tz=timezone.utc).isoformat()
    joined = attach_prediction_context(preds, samples)
    out = Path(f"artifacts/predictions/{model_id}.parquet")
    joined.to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
