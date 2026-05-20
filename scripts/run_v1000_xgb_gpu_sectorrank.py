"""H2 (GPU): XGBoost CUDA on sector-neutral rank target."""

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
from afp.targets.target_transform import TargetConfig, apply as apply_target


def main(model_id: str = "xgb_gpu_sectorrank"):
    samples = pd.read_parquet("data/processed/event_samples.parquet")
    facts = pd.read_parquet("data/processed/financial_facts_long.parquet")
    prices = pd.read_parquet("data/processed/prices_daily.parquet")
    sectors = pd.read_parquet("data/processed/sectors.parquet")
    encoder = AnonymousFeatureEncoder.load("artifacts/feature_encoder/v1")

    samples = samples.merge(
        sectors[["internal_company_id", "sic_division"]].rename(
            columns={"sic_division": "sector"}),
        on="internal_company_id", how="left",
    )
    samples = apply_target(samples, TargetConfig(target_mode="sector_neutral_rank", k=2.5))

    train = samples[samples["split"] == "train"].dropna(subset=["target_normalized_signal"])
    val = samples[samples["split"] == "validation"].dropna(subset=["target_normalized_signal"])
    test = samples[samples["split"] == "test"].dropna(subset=["target_normalized_signal"])
    print(f"sector-neutral rank target: train {len(train)}, val {len(val)}, test {len(test)}")

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

    model = build_model("xgb_gpu", {"params": {
        "n_estimators": 300, "learning_rate": 0.05, "max_depth": 6,
        "subsample": 0.9, "colsample_bytree": 0.5, "min_child_weight": 10,
        "device": "cuda", "tree_method": "hist",
    }})
    model.fit(train_ds.X, train_ds.y)
    val_pred = model.predict(val_ds.X)
    print("val metrics:", {k: round(v, 4) for k, v in regression_metrics(val_ds.y, val_pred).items()})

    test_pred = model.predict(test_ds.X)
    rows = []
    for sid, p in zip(test_ds.sample_ids, test_pred):
        rows.append({"sample_id": sid, "predicted_signal": float(p), "split": "test"})
    for sid, p in zip(val_ds.sample_ids, val_pred):
        rows.append({"sample_id": sid, "predicted_signal": float(p), "split": "validation"})
    preds = pd.DataFrame(rows)
    preds["model_id"] = model_id
    preds["model_type"] = "xgb_gpu_sectorrank"
    preds["prediction_created_at"] = datetime.now(tz=timezone.utc).isoformat()

    samples_for_join = samples.drop(columns=["sector"])
    joined = attach_prediction_context(preds, samples_for_join)
    out = Path(f"artifacts/predictions/{model_id}.parquet")
    joined.to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
