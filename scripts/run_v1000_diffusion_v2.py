"""Phase 23: diffusion v2 as the main model.

Improvements over v1:
  - Condition on fundamentals + price features (X augmented)
  - Bigger architecture (cond_hidden=1024, denoise=512)
  - More epochs (100)
  - More inference samples (128) for smoother distributional stats

Test data is held out from training. Validation rows flow only through inference.
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
from afp.models.diffusion import ConditionalDiffusion, DiffusionConfig
from afp.models.metrics import regression_metrics
from afp.models.train import attach_prediction_context


def main(model_id: str = "diffusion_v2", epochs: int = 100, inference_samples: int = 128,
         enable_price: bool = True):
    samples = pd.read_parquet("data/processed/event_samples.parquet")
    facts = pd.read_parquet("data/processed/financial_facts_long.parquet")
    prices = pd.read_parquet("data/processed/prices_daily.parquet")
    encoder = AnonymousFeatureEncoder.load("artifacts/feature_encoder/v1")

    train = samples[samples["split"] == "train"].dropna(subset=["target_normalized_signal"])
    val = samples[samples["split"] == "validation"].dropna(subset=["target_normalized_signal"])
    test = samples[samples["split"] == "test"].dropna(subset=["target_normalized_signal"])

    feature_ids = encoder.artifact.feature_map.feature_ids
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

    def _ds(samples_df, price_df):
        scaled, missing, meta, sids = encoder.transform(samples_df, facts)
        targets = samples_df.set_index("sample_id")["target_normalized_signal"]
        base = build_tabular_dataset(scaled, missing, meta, sids, targets, feature_ids)
        if enable_price:
            sub = price_df.set_index("sample_id").reindex(base.sample_ids)
            extra = sub[price_ids].to_numpy(dtype=np.float64)
            extra_miss = sub[[f"{fid}_missing" for fid in price_ids]].to_numpy(dtype=np.float64)
            base.X = np.concatenate([base.X, extra, extra_miss], axis=1)
        return base

    train_ds = _ds(train, price_train_df)
    val_ds = _ds(val, price_val_df)
    test_ds = _ds(test, price_test_df)
    print(f"X shapes — train: {train_ds.X.shape}, val: {val_ds.X.shape}, test: {test_ds.X.shape}")

    cfg = DiffusionConfig(
        epochs=epochs,
        batch_size=256,
        timesteps=200,
        cond_hidden=1024, cond_latent=192,
        denoise_hidden=512, time_embed_dim=64,
        inference_samples=inference_samples,
        lr=2e-4,
        dropout=0.3,
    )
    model = ConditionalDiffusion(cfg)
    print(f"fitting diffusion v2 (epochs={epochs}, cond_hidden=1024, denoise=512)...")
    model.fit(train_ds.X, train_ds.y)
    print("fit done. Running inference...")

    val_out = model.predict_distribution(val_ds.X)
    val_metrics = regression_metrics(val_ds.y, val_out["mean"])
    print("val metrics:", {k: round(v, 4) for k, v in val_metrics.items()})

    test_out = model.predict_distribution(test_ds.X)
    print(f"test mean range [{test_out['mean'].min():.3f}, {test_out['mean'].max():.3f}], "
          f"avg std {test_out['std'].mean():.3f}, "
          f"avg p_pos {test_out['probability_positive'].mean():.3f}")

    rows = []
    for ds, out, name in ((val_ds, val_out, "validation"),
                          (test_ds, test_out, "test")):
        for i, sid in enumerate(ds.sample_ids):
            rows.append({
                "sample_id": sid,
                "predicted_signal": float(out["mean"][i]),
                "predicted_signal_std": float(out["std"][i]),
                "probability_positive": float(out["probability_positive"][i]),
                "q05": float(out["q05"][i]),
                "q95": float(out["q95"][i]),
                "split": name,
            })
    preds = pd.DataFrame(rows)
    preds["model_id"] = model_id
    preds["model_type"] = "diffusion_v2"
    preds["prediction_created_at"] = datetime.now(tz=timezone.utc).isoformat()
    joined = attach_prediction_context(preds, samples)
    out_path = Path(f"artifacts/predictions/{model_id}.parquet")
    joined.to_parquet(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 100)
