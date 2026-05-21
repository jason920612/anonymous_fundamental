"""Phase 65: Apply Phase 62 LambdaRank-with-cross-features model to tier2.

Re-uses the tier2 sample-building pipeline but augments features
with tier2-computed cross-disciplinary market features. The model
is RE-TRAINED on tier1 train pool with cross features (same as
Phase 62) and applied to tier2 test rows.

Output: artifacts/predictions/lambdarank_tier2_cross.parquet
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
from afp.models.train import attach_prediction_context


def main():
    import lightgbm as lgb
    import time

    # ---------- tier1 train + val (with tier1 cross features) ----------
    samples_t1 = pd.read_parquet("data/processed/event_samples.parquet")
    facts_t1 = pd.read_parquet("data/processed/financial_facts_long.parquet")
    prices_t1 = pd.read_parquet("data/processed/prices_daily.parquet")
    encoder = AnonymousFeatureEncoder.load("artifacts/feature_encoder/v1")

    train = samples_t1[samples_t1["split"] == "train"].dropna(subset=["target_normalized_signal"])
    val = samples_t1[samples_t1["split"] == "validation"].dropna(subset=["target_normalized_signal"])

    # ---------- tier2 test ----------
    samples_t2 = pd.read_parquet("data/processed/tier2/event_samples.parquet")
    facts_t2 = pd.read_parquet("data/processed/tier2/financial_facts_long.parquet")
    prices_t2 = pd.read_parquet("data/processed/tier2/prices_daily.parquet")
    test_t2 = samples_t2[samples_t2["split"] == "test"].dropna(subset=["target_normalized_signal"])

    # ---------- Compute cross features per universe ----------
    print("computing tier1 cross features...", flush=True)
    pivot_t1 = (prices_t1.assign(date=pd.to_datetime(prices_t1["date"]))
                          .pivot_table(index="date", columns="internal_company_id",
                                       values="adjusted_close", aggfunc="last").ffill())
    cross_t1 = compute_cross_features(pivot_t1)
    print("computing tier2 cross features...", flush=True)
    pivot_t2 = (prices_t2.assign(date=pd.to_datetime(prices_t2["date"]))
                          .pivot_table(index="date", columns="internal_company_id",
                                       values="adjusted_close", aggfunc="last").ffill())
    cross_t2 = compute_cross_features(pivot_t2)

    train = attach_cross_features_to_samples(train, cross_t1)
    val = attach_cross_features_to_samples(val, cross_t1)
    test_t2 = attach_cross_features_to_samples(test_t2, cross_t2)

    # ---------- Feature pipeline ----------
    feature_ids = encoder.artifact.feature_map.feature_ids
    pf_cfg = PriceFeatureConfig(enabled=True)
    pf_train, price_ids = compute_price_features(train, prices_t1, pf_cfg)
    pf_val, _ = compute_price_features(val, prices_t1, pf_cfg)
    pf_test, _ = compute_price_features(test_t2, prices_t2, pf_cfg)
    stats = fit_scaler(pf_train, price_ids)
    pf_train = transform_with_scaler(pf_train, price_ids, stats)
    pf_val = transform_with_scaler(pf_val, price_ids, stats)
    pf_test = transform_with_scaler(pf_test, price_ids, stats)

    def _ds(samples_df, price_df, facts_to_use):
        scaled_, missing_, meta_, sids = encoder.transform(samples_df, facts_to_use)
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

    train_ds, train_rows = _ds(train, pf_train, facts_t1)
    val_ds, val_rows = _ds(val, pf_val, facts_t1)
    test_ds, test_rows = _ds(test_t2, pf_test, facts_t2)
    print(f"tier2 X shape with cross features: {test_ds.X.shape}", flush=True)

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
    print("training LambdaRank with cross features on tier1 pool...", flush=True)
    t0 = time.time()
    booster = lgb.train(params, train_data, num_boost_round=600,
                        valid_sets=[val_data],
                        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    print(f"  trained in {time.time()-t0:.1f}s", flush=True)

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
    preds["model_id"] = "lambdarank_tier2_cross"
    preds["model_type"] = "lambdarank_tier2_cross"
    preds["prediction_created_at"] = datetime.now(tz=timezone.utc).isoformat()
    joined = attach_prediction_context(preds, samples_t2)
    out = Path("artifacts/predictions/lambdarank_tier2_cross.parquet")
    joined.to_parquet(out, index=False)
    print(f"wrote {out} ({len(joined)} predictions)", flush=True)


if __name__ == "__main__":
    main()
