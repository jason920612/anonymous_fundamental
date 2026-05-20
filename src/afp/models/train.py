"""Train + predict orchestration for baseline models (RFC-05 §8-§11).

Phase 15: optional sample weighting (inverse-company-frequency × inverse-
quarter-frequency) to prevent mega-cap names like AAPL with 100+ filings
from dominating the regression / boosting loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from afp.features.encoder import AnonymousFeatureEncoder
from afp.models.baselines import BaseModel, build_model
from afp.models.datasets import TabularDataset, build_tabular_dataset
from afp.models.metrics import regression_metrics


def compute_sample_weights(
    train_samples: pd.DataFrame,
    sample_ids: np.ndarray,
    by_company: bool = True,
    by_quarter: bool = True,
) -> np.ndarray | None:
    """Inverse-frequency weights. Returns None if both flags are False.

    weight_i ∝ 1 / freq(company_i) × 1 / freq(quarter_i)
    Normalised to mean=1 so the effective dataset size is unchanged.
    """
    if not by_company and not by_quarter:
        return None
    idx = train_samples.set_index("sample_id")
    rows = idx.loc[sample_ids]
    w = np.ones(len(sample_ids), dtype=np.float64)
    if by_company:
        counts = rows["internal_company_id"].value_counts()
        w *= 1.0 / counts.reindex(rows["internal_company_id"].to_numpy()).to_numpy()
    if by_quarter and "entry_date" in rows.columns:
        q = pd.to_datetime(rows["entry_date"]).dt.to_period("Q").astype(str).to_numpy()
        qc = pd.Series(q).value_counts()
        w *= 1.0 / qc.reindex(q).to_numpy()
    w = w / w.mean()
    return w


@dataclass
class TrainResult:
    model: BaseModel
    model_id: str
    validation_metrics: dict[str, float]
    predictions: pd.DataFrame


def train_and_predict(
    model_kind: str,
    encoder: AnonymousFeatureEncoder,
    train_samples: pd.DataFrame,
    val_samples: pd.DataFrame,
    test_samples: pd.DataFrame,
    facts: pd.DataFrame,
    model_config: dict | None = None,
    model_id: str | None = None,
    weight_by_company: bool = False,
    weight_by_quarter: bool = False,
) -> TrainResult:
    """Fit `model_kind` on train, score val + test, return predictions for all rows."""
    feature_ids = encoder.artifact.feature_map.feature_ids
    # If the encoder produced derived columns, expand feature_ids to match
    if encoder.artifact.derived_enabled:
        feature_ids = (list(feature_ids)
                       + [f"derived_yoy_{fid}" for fid in feature_ids]
                       + [f"derived_z_{fid}" for fid in feature_ids])

    def _ds(samples: pd.DataFrame) -> TabularDataset:
        scaled, missing, meta, sids = encoder.transform(samples, facts)
        targets = samples.set_index("sample_id")["target_normalized_signal"]
        return build_tabular_dataset(scaled, missing, meta, sids, targets, feature_ids)

    train_ds = _ds(train_samples)
    val_ds = _ds(val_samples)
    test_ds = _ds(test_samples)

    sample_weight = compute_sample_weights(
        train_samples, train_ds.sample_ids,
        by_company=weight_by_company, by_quarter=weight_by_quarter,
    )
    model = build_model(model_kind, model_config or {})
    model.fit(train_ds.X, train_ds.y, sample_weight=sample_weight)

    val_pred = model.predict(val_ds.X)
    val_metrics = regression_metrics(val_ds.y, val_pred)

    rows = []
    for split_name, ds in (("validation", val_ds), ("test", test_ds)):
        if len(ds.sample_ids) == 0:
            continue
        preds = model.predict(ds.X)
        for sid, p in zip(ds.sample_ids, preds):
            rows.append({"sample_id": sid, "predicted_signal": float(p), "split": split_name})

    # Also score training for diagnostics
    train_pred = model.predict(train_ds.X)
    for sid, p in zip(train_ds.sample_ids, train_pred):
        rows.append({"sample_id": sid, "predicted_signal": float(p), "split": "train"})

    predictions = pd.DataFrame(rows)
    predictions["model_id"] = model_id or f"{model.model_type}_v001"
    predictions["model_type"] = model.model_type
    predictions["prediction_created_at"] = datetime.now(tz=timezone.utc).isoformat()
    return TrainResult(model=model, model_id=predictions["model_id"].iloc[0],
                       validation_metrics=val_metrics, predictions=predictions)


def attach_prediction_context(predictions: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Add `entry_date`, `exit_date`, `internal_company_id`, `ex_ante_scale` to predictions
    so the portfolio module can restore returns without re-joining."""
    keep = ["sample_id", "internal_company_id", "entry_date", "exit_date",
            "ex_ante_scale", "target_normalized_signal", "raw_log_return"]
    return predictions.merge(samples[keep], on="sample_id", how="left")
