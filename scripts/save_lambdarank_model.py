"""Train LambdaRank tuned on the original 1000-CIK train pool and SAVE the booster
so the predict CLI can load it without retraining.

Output:
  artifacts/models/lambdarank_v3/booster.txt    LightGBM dump
  artifacts/models/lambdarank_v3/metadata.json  feature counts, price stats
  artifacts/models/lambdarank_v3/price_stats.npz  per-feature scaler stats
"""

import json
import sys
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


def main():
    import lightgbm as lgb

    samples = pd.read_parquet("data/processed/event_samples.parquet")
    facts = pd.read_parquet("data/processed/financial_facts_long.parquet")
    prices = pd.read_parquet("data/processed/prices_daily.parquet")
    encoder = AnonymousFeatureEncoder.load("artifacts/feature_encoder/v1")

    train = samples[samples["split"] == "train"].dropna(subset=["target_normalized_signal"])
    val = samples[samples["split"] == "validation"].dropna(subset=["target_normalized_signal"])

    feature_ids = encoder.artifact.feature_map.feature_ids
    pf_cfg = PriceFeatureConfig(enabled=True)
    pf_train, price_ids = compute_price_features(train, prices, pf_cfg)
    pf_val, _ = compute_price_features(val, prices, pf_cfg)
    price_stats = fit_scaler(pf_train, price_ids)
    pf_train = transform_with_scaler(pf_train, price_ids, price_stats)
    pf_val = transform_with_scaler(pf_val, price_ids, price_stats)

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
        "max_position": 50, "label_gain": [pow(2, i) - 1 for i in range(31)],
    }
    print("training LambdaRank tuned (for save)...")
    booster = lgb.train(params, train_data, num_boost_round=600,
                        valid_sets=[val_data],
                        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

    out = Path("artifacts/models/lambdarank_v3")
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "booster.txt"))

    # Save the price-feature scaler stats so predict can re-apply
    np.savez(out / "price_stats.npz",
             feature_ids=np.array(price_ids),
             medians=np.array([price_stats[fid][0] for fid in price_ids]),
             iqrs=np.array([price_stats[fid][1] for fid in price_ids]))

    meta = {
        "feature_encoder_dir": "artifacts/feature_encoder/v1",
        "n_base_features": len(feature_ids),
        "price_feature_ids": list(price_ids),
        "best_iteration": int(booster.best_iteration or 0),
        "params": params,
        "trained_on": "1000-CIK train pool, 2003-2015",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"saved booster + metadata to {out}")


if __name__ == "__main__":
    main()
