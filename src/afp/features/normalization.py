"""Signed-log + robust-scale value transform (RFC-03 §8)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NormalizerConfig:
    clip_min: float = -10.0
    clip_max: float = 10.0
    epsilon: float = 1.0e-6


def signed_log(x: np.ndarray) -> np.ndarray:
    """`sign(x) * log1p(|x|)` — handles ±, zero, large magnitudes."""
    return np.sign(x) * np.log1p(np.abs(x))


class RobustScaler:
    """Per-feature, per-period median / IQR scaler.

    Fit only on training data (RFC-03 §13). Stores median/IQR with shape
    `[periods_back, n_features]` so each (offset, feature) cell has its own stats.
    """

    def __init__(self, config: NormalizerConfig | None = None) -> None:
        self.config = config or NormalizerConfig()
        self.median: np.ndarray | None = None     # [P, F]
        self.iqr: np.ndarray | None = None        # [P, F]

    def fit(self, values: np.ndarray, missing: np.ndarray) -> "RobustScaler":
        signed = signed_log(np.where(missing == 0, values, np.nan))
        P, F = signed.shape[1], signed.shape[2]
        median = np.zeros((P, F), dtype=np.float64)
        iqr = np.zeros((P, F), dtype=np.float64)
        for p in range(P):
            for f in range(F):
                col = signed[:, p, f]
                col = col[~np.isnan(col)]
                if col.size == 0:
                    continue
                median[p, f] = float(np.median(col))
                q75 = float(np.percentile(col, 75))
                q25 = float(np.percentile(col, 25))
                iqr[p, f] = q75 - q25
        self.median = median
        self.iqr = iqr
        return self

    def transform(self, values: np.ndarray, missing: np.ndarray) -> np.ndarray:
        if self.median is None or self.iqr is None:
            raise RuntimeError("RobustScaler not fit")
        signed = signed_log(np.where(missing == 0, values, 0.0))
        denom = np.where(self.iqr > 0, self.iqr, self.config.epsilon)
        scaled = (signed - self.median[None, :, :]) / denom[None, :, :]
        scaled = np.where(missing == 0, scaled, 0.0)
        return np.clip(scaled, self.config.clip_min, self.config.clip_max)
