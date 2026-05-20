"""Phase 20 — ensemble averaging."""

import numpy as np
import pandas as pd

from afp.models.ensemble import equal_weight_ensemble, rank_average_ensemble


def _write_preds(tmp_path, name, signals, split="test"):
    df = pd.DataFrame({
        "sample_id": [f"S{i}" for i in range(len(signals))],
        "predicted_signal": signals,
        "split": split,
        "model_id": name,
        "model_type": "test",
    })
    path = tmp_path / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def test_mean_ensemble_averages_two_models(tmp_path):
    a = _write_preds(tmp_path, "a", np.array([0.5, 0.0, -0.5]))
    b = _write_preds(tmp_path, "b", np.array([0.1, 0.2, 0.3]))
    out = tmp_path / "ens.parquet"
    equal_weight_ensemble([a, b], out)
    df = pd.read_parquet(out)
    assert np.allclose(df["predicted_signal"], [0.30, 0.10, -0.10])


def test_rank_ensemble_centered_at_zero(tmp_path):
    rng = np.random.default_rng(0)
    a = _write_preds(tmp_path, "a", rng.normal(size=200))
    b = _write_preds(tmp_path, "b", rng.normal(size=200))
    out = tmp_path / "ens.parquet"
    rank_average_ensemble([a, b], out)
    df = pd.read_parquet(out)
    # Rank avg ∈ [-1, 1], centered ~0
    assert (df["predicted_signal"] >= -1).all() and (df["predicted_signal"] <= 1).all()
    assert abs(df["predicted_signal"].mean()) < 0.05
