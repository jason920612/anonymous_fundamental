"""James-Stein signal shrinkage (Phase 48).

For N ≥ 3 means estimated from noisy observations, the James-Stein
estimator strictly dominates the MLE in mean-squared error (Stein
1956; James-Stein 1961):

    μ_JS_i = μ_grand + (1 - c) · (μ_i - μ_grand)

where the shrinkage factor `c ∈ [0, 1]` is:

    c = min(1, (N - 2) · σ² / Σ_i (μ_i - μ_grand)²)

with `σ²` the variance of the noise around each `μ_i`. The result is
that every μ_i is "shrunk" toward the cross-sectional mean by a
data-determined factor.

This module provides a wrapper that pre-processes the candidate
signal vector before it reaches the allocator. The wrapper is
parameter-free for the shrinkage factor itself — `σ²` is estimated
from the candidate signals' cross-sectional variance as a noise
proxy. Theory-driven; not tuned on val/test.
"""

from __future__ import annotations

import numpy as np


def js_shrink(signal: np.ndarray, noise_var: float | None = None) -> np.ndarray:
    """Apply James-Stein shrinkage to a signal vector toward its grand mean.

    Parameters
    ----------
    signal : (N,) signal vector. Higher = more positive.
    noise_var : optional explicit noise variance. If None, uses the
        cross-sectional variance / N as a proxy — a conservative
        choice that produces stable shrinkage.

    Returns
    -------
    (N,) shrunken signal with the same grand mean as the input.
    """
    n = len(signal)
    if n < 3:
        return signal.copy()
    mu_grand = float(signal.mean())
    deviations = signal - mu_grand
    ss = float((deviations ** 2).sum())
    if ss <= 1e-12:
        return signal.copy()
    if noise_var is None:
        noise_var = float(signal.var(ddof=1)) / n
    if noise_var <= 0:
        return signal.copy()
    c = min(1.0, max(0.0, (n - 2) * noise_var / ss))
    return mu_grand + (1.0 - c) * deviations
