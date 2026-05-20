"""Phase 20: simple equal-weight ensemble of predictions parquets.

Useful because Ridge, LightGBM, and XGBoost express different inductive biases
on the same anonymous features. With Spearman in the 0.02-0.05 range per
model, averaging often reduces variance enough to lift portfolio Sharpe
without changing the input representation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def equal_weight_ensemble(prediction_paths: list[str | Path], out_path: str | Path,
                          model_id: str = "ensemble_v1") -> Path:
    frames = []
    for p in prediction_paths:
        df = pd.read_parquet(p)
        df = df[["sample_id", "predicted_signal", "split"]]
        df = df.rename(columns={"predicted_signal": f"pred_{Path(p).stem}"})
        frames.append(df)
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on=["sample_id", "split"], how="inner")
    pred_cols = [c for c in merged.columns if c.startswith("pred_")]
    merged["predicted_signal"] = merged[pred_cols].mean(axis=1).clip(-1, 1)
    merged["model_id"] = model_id
    merged["model_type"] = "ensemble"
    out = merged[["sample_id", "predicted_signal", "split", "model_id", "model_type"]]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out_path


def rank_average_ensemble(prediction_paths: list[str | Path], out_path: str | Path,
                          model_id: str = "ensemble_rank_v1",
                          split_for_rank: str = "validation") -> Path:
    """Rank-average within each split — more robust to scale differences across models."""
    frames = []
    for p in prediction_paths:
        df = pd.read_parquet(p)[["sample_id", "predicted_signal", "split"]]
        df["rank"] = df.groupby("split")["predicted_signal"].rank(pct=True)
        df = df.rename(columns={"rank": f"rank_{Path(p).stem}"})
        df = df.drop(columns=["predicted_signal"])
        frames.append(df)
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on=["sample_id", "split"], how="inner")
    rank_cols = [c for c in merged.columns if c.startswith("rank_")]
    merged["predicted_signal"] = (2.0 * merged[rank_cols].mean(axis=1) - 1.0).clip(-1, 1)
    merged["model_id"] = model_id
    merged["model_type"] = "ensemble_rank"
    out = merged[["sample_id", "predicted_signal", "split", "model_id", "model_type"]]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out_path
