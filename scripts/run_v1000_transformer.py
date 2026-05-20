"""H4 (GPU): tabular transformer over period axis."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afp.features.encoder import AnonymousFeatureEncoder
from afp.models.datasets import build_tabular_dataset
from afp.models.metrics import regression_metrics
from afp.models.period_transformer import PeriodTransformer, PeriodTransformerConfig
from afp.models.train import attach_prediction_context


def main(model_id: str = "transformer_v1000", epochs: int = 60):
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
    print(f"train X: {train_ds.X.shape}, feature count F: {len(feature_ids)}")

    cfg = PeriodTransformerConfig(d_model=256, n_heads=8, n_layers=3,
                                  dropout=0.2, batch_size=256, epochs=epochs, lr=3e-4)
    model = PeriodTransformer(cfg)
    print(f"fitting transformer (epochs={epochs}, d_model={cfg.d_model})...")
    model.fit(train_ds.X, train_ds.y, feature_count=len(feature_ids))

    val_pred = model.predict(val_ds.X)
    print("val metrics:", {k: round(v, 4) for k, v in regression_metrics(val_ds.y, val_pred).items()})

    test_pred = model.predict(test_ds.X)
    rows = []
    for sid, p in zip(val_ds.sample_ids, val_pred):
        rows.append({"sample_id": sid, "predicted_signal": float(p), "split": "validation"})
    for sid, p in zip(test_ds.sample_ids, test_pred):
        rows.append({"sample_id": sid, "predicted_signal": float(p), "split": "test"})
    preds = pd.DataFrame(rows)
    preds["model_id"] = model_id
    preds["model_type"] = "period_transformer"
    preds["prediction_created_at"] = datetime.now(tz=timezone.utc).isoformat()
    joined = attach_prediction_context(preds, samples)
    out = Path(f"artifacts/predictions/{model_id}.parquet")
    joined.to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 60)
