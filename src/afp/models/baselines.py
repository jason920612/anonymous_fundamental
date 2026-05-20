"""Baseline regressors (RFC-05 §4): constant zero, historical mean, ridge, GBT.

Phase 19 adds GPU-accelerated `xgb_gpu` (XGBoost CUDA) and `mlp_gpu`
(PyTorch on CUDA when available). LightGBM GPU support requires a custom
wheel and is not assumed — `gbt` still falls back to LightGBM CPU or
sklearn HistGradientBoosting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


class BaseModel(ABC):
    """All baselines accept dense matrices and emit predictions clipped to [-1,1]."""

    model_type: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    def _clip(p: np.ndarray) -> np.ndarray:
        return np.clip(p, -1.0, 1.0)


class ConstantZero(BaseModel):
    model_type = "constant_zero"

    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=np.float64)


class HistoricalMean(BaseModel):
    model_type = "historical_mean"

    def __init__(self):
        self.mean_ = 0.0

    def fit(self, X, y, sample_weight=None):
        if len(y) == 0:
            self.mean_ = 0.0
        elif sample_weight is not None:
            self.mean_ = float(np.average(y, weights=sample_weight))
        else:
            self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return self._clip(np.full(len(X), self.mean_, dtype=np.float64))


class RidgeModel(BaseModel):
    model_type = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = None

    def fit(self, X, y, sample_weight=None):
        self.model = Ridge(alpha=self.alpha).fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self._clip(self.model.predict(X))


class GradientBoostedTrees(BaseModel):
    """Tries LightGBM if available, otherwise sklearn HistGradientBoostingRegressor."""

    model_type = "gbt"

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self.model = None
        self._engine = "hist_gbt"

    def fit(self, X, y, sample_weight=None):
        try:
            import lightgbm as lgb
            self._engine = "lightgbm"
            self.model = lgb.LGBMRegressor(**{
                "objective": "regression",
                "n_estimators": 500,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_data_in_leaf": 20,
                "verbose": -1,
                **self.params,
            }).fit(X, y, sample_weight=sample_weight)
        except ImportError:
            self._engine = "hist_gbt"
            self.model = HistGradientBoostingRegressor(**{
                "loss": "squared_error",
                "max_iter": 500,
                "learning_rate": 0.05,
                "max_depth": 6,
                **self.params,
            }).fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self._clip(self.model.predict(X))


class XGBoostGPU(BaseModel):
    """XGBoost histogram tree booster running on CUDA."""

    model_type = "xgb_gpu"

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self.model = None

    def fit(self, X, y, sample_weight=None):
        import xgboost as xgb
        defaults = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": "cuda",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 10.0,
            "subsample": 0.9,
            "colsample_bytree": 0.5,
            "verbosity": 0,
        }
        defaults.update(self.params)
        self.model = xgb.XGBRegressor(**defaults).fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self._clip(self.model.predict(X))


class MLPGPU(BaseModel):
    """Small dense MLP regressor on PyTorch / CUDA when available.

    Architecture: Linear → GELU → Dropout → Linear → GELU → Dropout → Linear → tanh.
    Loss: Huber. Optimizer: AdamW with cosine decay.
    """

    model_type = "mlp_gpu"

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self.model = None
        self.device = None
        self.mean_ = 0.0
        self.std_ = 1.0

    def _build(self, n_features: int):
        from torch import nn
        h1 = int(self.params.get("hidden_1", 512))
        h2 = int(self.params.get("hidden_2", 128))
        dropout = float(self.params.get("dropout", 0.3))
        return nn.Sequential(
            nn.Linear(n_features, h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h2, 1), nn.Tanh(),
        )

    def fit(self, X, y, sample_weight=None):
        import torch
        from torch import nn
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_features = X.shape[1]
        self.model = self._build(n_features).to(self.device)

        seed = int(self.params.get("seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)

        X = np.nan_to_num(X.astype(np.float32))
        y = np.nan_to_num(y.astype(np.float32))
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-6
        Xs = (X - self.mean_) / self.std_

        X_t = torch.from_numpy(Xs).to(self.device)
        y_t = torch.from_numpy(y).to(self.device).unsqueeze(1)
        w_t = (torch.from_numpy(sample_weight.astype(np.float32)).to(self.device).unsqueeze(1)
               if sample_weight is not None else None)

        batch_size = int(self.params.get("batch_size", 512))
        epochs = int(self.params.get("epochs", 80))
        lr = float(self.params.get("lr", 1e-3))
        weight_decay = float(self.params.get("weight_decay", 1e-4))
        opt = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.SmoothL1Loss(reduction="none", beta=0.1)

        n = len(X_t)
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            rng.shuffle(idx)
            self.model.train()
            for start in range(0, n, batch_size):
                batch = idx[start:start + batch_size]
                xb = X_t[batch]; yb = y_t[batch]
                pred = self.model(xb)
                l = loss_fn(pred, yb)
                if w_t is not None:
                    l = (l * w_t[batch]).mean()
                else:
                    l = l.mean()
                opt.zero_grad()
                l.backward()
                opt.step()
            sched.step()
        return self

    def predict(self, X):
        import torch
        X = np.nan_to_num(X.astype(np.float32))
        Xs = (X - self.mean_) / self.std_
        self.model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(Xs).to(self.device)
            out_chunks = []
            chunk = 8192
            for i in range(0, len(X_t), chunk):
                out_chunks.append(self.model(X_t[i:i + chunk]).cpu().numpy().ravel())
        return self._clip(np.concatenate(out_chunks))


# ---------------------------------------------------------------------------

def build_model(kind: str, config: dict | None = None) -> BaseModel:
    config = config or {}
    if kind == "constant_zero":
        return ConstantZero()
    if kind == "historical_mean":
        return HistoricalMean()
    if kind == "ridge":
        return RidgeModel(alpha=config.get("alpha", 1.0))
    if kind in ("lightgbm", "hist_gbt", "gbt"):
        return GradientBoostedTrees(params=config.get("params", {}))
    if kind in ("xgb_gpu", "xgboost_gpu"):
        return XGBoostGPU(params=config.get("params", config))
    if kind in ("mlp_gpu", "mlp"):
        return MLPGPU(params=config.get("params", config))
    raise ValueError(f"unknown model kind {kind!r}")
