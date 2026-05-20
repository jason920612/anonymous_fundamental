"""Phase 22: train conditional diffusion on v1000 anonymous features, predict on val+test.

GPU when available. Test data is held out from fit.
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
from afp.models.datasets import build_tabular_dataset
from afp.models.diffusion import ConditionalDiffusion, DiffusionConfig
from afp.models.metrics import regression_metrics
from afp.models.train import attach_prediction_context


def main(model_id: str = "diffusion_v1000", epochs: int = 60, inference_samples: int = 64):
    samples = pd.read_parquet("data/processed/event_samples.parquet")
    facts = pd.read_parquet("data/processed/financial_facts_long.parquet")
    encoder = AnonymousFeatureEncoder.load("artifacts/feature_encoder/v1")

    train = samples[samples["split"] == "train"].dropna(subset=["target_normalized_signal"])
    val = samples[samples["split"] == "validation"].dropna(subset=["target_normalized_signal"])
    test = samples[samples["split"] == "test"].dropna(subset=["target_normalized_signal"])

    feature_ids = encoder.artifact.feature_map.feature_ids

    def _ds(rows):
        scaled, missing, meta, sids = encoder.transform(rows, facts)
        targets = rows.set_index("sample_id")["target_normalized_signal"]
        return build_tabular_dataset(scaled, missing, meta, sids, targets, feature_ids)

    train_ds = _ds(train); val_ds = _ds(val); test_ds = _ds(test)
    print(f"train X: {train_ds.X.shape}, val: {val_ds.X.shape}, test: {test_ds.X.shape}")

    cfg = DiffusionConfig(
        epochs=epochs,
        batch_size=256,
        timesteps=200,
        cond_hidden=512, cond_latent=128,
        denoise_hidden=256, time_embed_dim=64,
        inference_samples=inference_samples,
        lr=3e-4,
        dropout=0.3,
    )
    model = ConditionalDiffusion(cfg)
    print(f"fitting diffusion (epochs={epochs})...")
    model.fit(train_ds.X, train_ds.y)
    print("fit done. Running inference...")

    val_out = model.predict_distribution(val_ds.X)
    val_metrics = regression_metrics(val_ds.y, val_out["mean"])
    print("val metrics:", {k: round(v, 4) for k, v in val_metrics.items()})

    test_out = model.predict_distribution(test_ds.X)
    print(f"test mean predictions in range [{test_out['mean'].min():.3f}, {test_out['mean'].max():.3f}]")

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
    preds["model_type"] = "diffusion"
    preds["prediction_created_at"] = datetime.now(tz=timezone.utc).isoformat()
    joined = attach_prediction_context(preds, samples)
    out_path = Path(f"artifacts/predictions/{model_id}.parquet")
    joined.to_parquet(out_path, index=False)
    print(f"wrote {out_path}")
    return val_metrics


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    main(epochs=epochs)
