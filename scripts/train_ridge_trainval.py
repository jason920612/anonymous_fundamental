"""Train Ridge on train+val combined, score on test.

Test integrity preserved — test rows never enter the fit. This is the
standard "use validation for HPO/final-fit" pattern.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from afp.features.encoder import AnonymousFeatureEncoder
from afp.models.baselines import build_model
from afp.models.datasets import build_tabular_dataset
from afp.models.metrics import regression_metrics
from afp.models.train import attach_prediction_context


def main():
    samples = pd.read_parquet("data/processed/event_samples.parquet")
    facts = pd.read_parquet("data/processed/financial_facts_long.parquet")
    encoder = AnonymousFeatureEncoder.load("artifacts/feature_encoder/v1")

    train_val = samples[samples["split"].isin(("train", "validation"))].dropna(
        subset=["target_normalized_signal"]
    )
    test = samples[samples["split"] == "test"].dropna(subset=["target_normalized_signal"])
    print(f"train+val rows: {len(train_val)}, test rows: {len(test)}")

    feature_ids = encoder.artifact.feature_map.feature_ids

    def _ds(rows):
        scaled, missing, meta, sids = encoder.transform(rows, facts)
        targets = rows.set_index("sample_id")["target_normalized_signal"]
        return build_tabular_dataset(scaled, missing, meta, sids, targets, feature_ids)

    train_ds = _ds(train_val)
    test_ds = _ds(test)

    model = build_model("ridge", {"alpha": 1.0})
    model.fit(train_ds.X, train_ds.y)

    test_pred = model.predict(test_ds.X)
    metrics = regression_metrics(test_ds.y, test_pred)
    print("test metrics:", {k: round(v, 4) for k, v in metrics.items()})

    rows = []
    for sid, p in zip(test_ds.sample_ids, test_pred):
        rows.append({"sample_id": sid, "predicted_signal": float(p), "split": "test"})
    preds = pd.DataFrame(rows)
    preds["model_id"] = "ridge_trainval"
    preds["model_type"] = "ridge"
    preds["prediction_created_at"] = datetime.now(tz=timezone.utc).isoformat()
    joined = attach_prediction_context(preds, samples)
    out = Path("artifacts/predictions/ridge_trainval.parquet")
    joined.to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
