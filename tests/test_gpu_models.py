"""Phase 19 — GPU model adapters (skipped when GPU unavailable)."""

import numpy as np
import pytest

try:
    import xgboost  # noqa: F401
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

try:
    import torch
    HAVE_TORCH = torch.cuda.is_available() if hasattr(torch, "cuda") else False
except ImportError:
    HAVE_TORCH = False

from afp.models.baselines import build_model


@pytest.mark.skipif(not HAVE_XGB, reason="xgboost not installed")
def test_xgb_gpu_trains_and_predicts_in_range():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 30))
    y = X[:, 0] + rng.normal(0, 0.1, 400)
    m = build_model("xgb_gpu", {"params": {"n_estimators": 30, "device": "cuda"}})
    try:
        m.fit(X, y)
    except Exception as e:
        pytest.skip(f"GPU XGBoost unavailable in this env: {e}")
    pred = m.predict(X)
    assert pred.shape == (400,)
    assert ((pred >= -1) & (pred <= 1)).all()


@pytest.mark.skipif(not HAVE_TORCH, reason="torch CUDA unavailable")
def test_mlp_gpu_trains_and_predicts_in_range():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 30)).astype(np.float32)
    y = X[:, 0] + rng.normal(0, 0.1, 400).astype(np.float32)
    m = build_model("mlp_gpu", {"params": {"epochs": 5, "batch_size": 64, "hidden_1": 64, "hidden_2": 32}})
    m.fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (400,)
    assert ((pred >= -1) & (pred <= 1)).all()
