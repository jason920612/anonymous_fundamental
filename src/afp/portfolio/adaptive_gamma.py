"""Adaptive risk-aversion γ via Shannon entropy of the signal (Phase 49).

Theory: the optimal risk-aversion in Markowitz max(wᵀμ − γ/2 wᵀΣw)
depends on how concentrated the signal is. If μ has one clear
winner (low entropy) the optimum tilts heavily toward that name —
γ should be small. If μ is nearly uniform (high entropy) there is
no rational basis for concentration — γ should be large to fall
back on min-variance.

Implementation: at each allocation step, compute the Shannon entropy
of the normalized positive signal vector

    H(p) = -Σ p_i log p_i,   p_i = μ_i / Σ_j μ_j

Normalize to [0, 1] by dividing by log(N), then map linearly to a
pre-declared γ range:

    γ(H) = γ_min + (γ_max - γ_min) · H_norm

With γ_min=2, γ_max=10. These are the **same** values used in the
Phase 41 γ-sensitivity check, treated here as a structural envelope.
No fit to val/test data.
"""

from __future__ import annotations

import numpy as np


def signal_entropy(signal: np.ndarray) -> float:
    """Shannon entropy of a non-negative signal as a probability mass.

    Returns 0 for any pathological input (negative, NaN, all-zero).
    """
    sig = np.asarray(signal, dtype=float)
    sig = np.clip(sig, 0.0, None)
    s = sig.sum()
    if s <= 1e-12 or not np.isfinite(s):
        return 0.0
    p = sig / s
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


def adaptive_gamma(signal: np.ndarray, gamma_min: float = 2.0,
                   gamma_max: float = 10.0) -> float:
    """Map signal entropy to a risk-aversion in [gamma_min, gamma_max]."""
    n = len(signal)
    if n <= 1:
        return gamma_max
    H = signal_entropy(signal)
    H_max = np.log(n)
    if H_max <= 1e-12:
        return gamma_max
    H_norm = float(np.clip(H / H_max, 0.0, 1.0))
    return gamma_min + (gamma_max - gamma_min) * H_norm
