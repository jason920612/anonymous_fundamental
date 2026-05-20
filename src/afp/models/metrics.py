"""Model-level evaluation metrics (RFC-05 §13)."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    out: dict[str, float] = {
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
        "direction_accuracy": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
    }
    if y_pred.std() > 0 and y_true.std() > 0:
        try:
            r_p, _ = pearsonr(y_pred, y_true)
            r_s, _ = spearmanr(y_pred, y_true)
            out["pearson"] = float(r_p)
            out["spearman"] = float(r_s)
        except Exception:
            out["pearson"] = float("nan")
            out["spearman"] = float("nan")
    else:
        out["pearson"] = 0.0
        out["spearman"] = 0.0

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) or 1e-12
    out["r2"] = 1.0 - ss_res / ss_tot
    return out


def bucket_report(y_true: np.ndarray, y_pred: np.ndarray,
                  num_buckets: int = 10) -> list[dict]:
    """Bucket predictions into deciles of `y_pred` and report mean true value per bucket."""
    if len(y_pred) == 0:
        return []
    edges = np.linspace(-1.0, 1.0, num_buckets + 1)
    buckets = []
    for i in range(num_buckets):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_pred >= lo) & (y_pred < hi if i < num_buckets - 1 else y_pred <= hi)
        if mask.sum() == 0:
            continue
        buckets.append({
            "bucket_lo": float(lo),
            "bucket_hi": float(hi),
            "count": int(mask.sum()),
            "mean_predicted": float(y_pred[mask].mean()),
            "mean_true": float(y_true[mask].mean()),
        })
    return buckets
