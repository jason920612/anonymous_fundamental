"""Phase 21 runner: train Ridge with anonymous fundamentals + anonymous price features.

Test data is held out: scalers / feature stats are fit on train only.
"""

import json
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
from afp.models.baselines import build_model
from afp.models.datasets import build_tabular_dataset
from afp.models.metrics import regression_metrics
from afp.models.train import attach_prediction_context


def main(model_kind: str = "ridge",
         model_id: str = "ridge_price_v1000",
         enable_price: bool = True):
    samples = pd.read_parquet("data/processed/event_samples.parquet")
    facts = pd.read_parquet("data/processed/financial_facts_long.parquet")
    prices = pd.read_parquet("data/processed/prices_daily.parquet")
    encoder = AnonymousFeatureEncoder.load("artifacts/feature_encoder/v1")

    train = samples[samples["split"] == "train"].dropna(subset=["target_normalized_signal"])
    val = samples[samples["split"] == "validation"].dropna(subset=["target_normalized_signal"])
    test = samples[samples["split"] == "test"].dropna(subset=["target_normalized_signal"])

    feature_ids = encoder.artifact.feature_map.feature_ids

    # Anonymous price features
    pf_cfg = PriceFeatureConfig(enabled=enable_price)
    print(f"computing price features for {len(samples)} samples...")
    price_train_df, price_ids = compute_price_features(train, prices, pf_cfg)
    price_val_df, _ = compute_price_features(val, prices, pf_cfg)
    price_test_df, _ = compute_price_features(test, prices, pf_cfg)
    if enable_price:
        stats = fit_scaler(price_train_df, price_ids)
        price_train_df = transform_with_scaler(price_train_df, price_ids, stats)
        price_val_df = transform_with_scaler(price_val_df, price_ids, stats)
        price_test_df = transform_with_scaler(price_test_df, price_ids, stats)
        print(f"price features: {len(price_ids)} new columns")

    def _ds(samples_df, price_df):
        scaled, missing, meta, sids = encoder.transform(samples_df, facts)
        targets = samples_df.set_index("sample_id")["target_normalized_signal"]
        base_ds = build_tabular_dataset(scaled, missing, meta, sids, targets, feature_ids)
        if enable_price:
            price_df_sub = price_df.set_index("sample_id").reindex(base_ds.sample_ids)
            extra = price_df_sub[price_ids].to_numpy(dtype=np.float64)
            extra_miss = price_df_sub[[f"{fid}_missing" for fid in price_ids]].to_numpy(dtype=np.float64)
            base_ds.X = np.concatenate([base_ds.X, extra, extra_miss], axis=1)
        return base_ds

    train_ds = _ds(train, price_train_df)
    val_ds = _ds(val, price_val_df)
    test_ds = _ds(test, price_test_df)

    print(f"train X shape: {train_ds.X.shape}")
    model = build_model(model_kind, {"alpha": 1.0})
    model.fit(train_ds.X, train_ds.y)

    val_pred = model.predict(val_ds.X)
    val_metrics = regression_metrics(val_ds.y, val_pred)
    print("val metrics:", {k: round(v, 4) for k, v in val_metrics.items()})

    test_pred = model.predict(test_ds.X)
    rows = []
    for sid, p in zip(test_ds.sample_ids, test_pred):
        rows.append({"sample_id": sid, "predicted_signal": float(p), "split": "test"})
    for sid, p in zip(val_ds.sample_ids, val_pred):
        rows.append({"sample_id": sid, "predicted_signal": float(p), "split": "validation"})
    preds = pd.DataFrame(rows)
    preds["model_id"] = model_id
    preds["model_type"] = model_kind
    preds["prediction_created_at"] = datetime.now(tz=timezone.utc).isoformat()
    joined = attach_prediction_context(preds, samples)
    out = Path(f"artifacts/predictions/{model_id}.parquet")
    joined.to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "ridge"
    mid = sys.argv[2] if len(sys.argv) > 2 else f"{kind}_price_v1000"
    main(model_kind=kind, model_id=mid)
