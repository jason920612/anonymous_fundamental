"""Kuramoto synchronization order parameter (Phase 61).

Kuramoto 1975 — N coupled oscillators with natural frequencies and
sinusoidal coupling. The dynamics are characterized by the complex
order parameter

    r·e^(iψ) = (1/N) Σ_j e^(iφ_j)

where r ∈ [0, 1] measures phase coherence:

- r = 0: oscillators uniformly spread in phase (disordered).
- r = 1: all oscillators synchronized at common phase.

Above a critical coupling K_c, the system spontaneously synchronizes.

Applied to stock returns:

- Define each name's "phase" via the angle of its (cumulative return,
  vol) vector — or simply the SIGN/MAGNITUDE of its recent cumulative
  return relative to its rolling vol.
- We use a particularly simple proxy: φ_j(t) = π · sign(ret_j(t)) ·
  min(1, |ret_j(t)|/σ_j) — angle saturates at ±π based on rolling
  z-score of the day's return.
- Compute order parameter r(t) across the active universe.
- A rising r (synchronization growing) means market is in lockstep
  regime — selection alpha cannot escape the common factor — de-risk.

The detector is *output-side* in the sense that it uses recent
realized returns, not predicted signals. Pre-declared parameters
from Kuramoto theory; no tuning on val/test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KuramotoConfig:
    enabled: bool = False
    sigma_window: int = 60       # rolling vol per stock
    history_window: int = 252    # rolling history for r z-score
    boltzmann_kt: float = 2.0    # damping strength
    scale_floor: float = 0.30
    scale_ceiling: float = 1.0
    blend: float = 0.5


def _kuramoto_r(returns_today: np.ndarray, sigma_today: np.ndarray) -> float:
    """Order parameter from today's per-asset returns and rolling σ's."""
    sigma = np.where(sigma_today > 1e-9, sigma_today, 1.0)
    z = np.clip(returns_today / sigma, -1.0, 1.0)
    # Map z ∈ [-1, 1] to phase φ ∈ [-π, π]
    phi = np.pi * z
    e_iphi = np.exp(1j * phi)
    r = float(np.abs(e_iphi.mean()))
    return r


def compute_kuramoto_r_series(price_panel: pd.DataFrame,
                              sigma_window: int = 60) -> pd.Series:
    """Daily Kuramoto order parameter r(t) over the universe."""
    log_p = np.log(price_panel.astype(float).replace(0.0, np.nan))
    ret = log_p.diff()
    sigma = ret.rolling(sigma_window, min_periods=20).std(ddof=1)

    r_values = []
    for idx in ret.index:
        row = ret.loc[idx].dropna()
        if len(row) < 10:
            r_values.append(np.nan)
            continue
        s_row = sigma.loc[idx].reindex(row.index)
        if s_row.isna().any():
            r_values.append(np.nan)
            continue
        r_values.append(_kuramoto_r(row.to_numpy(), s_row.to_numpy()))
    return pd.Series(r_values, index=ret.index)


def compute_kuramoto_scale(r_series: pd.Series, cfg: KuramotoConfig,
                           asof: pd.Timestamp, prev_scale: float | None = None) -> float:
    if not cfg.enabled or r_series.empty:
        return 1.0
    past = r_series.loc[r_series.index <= asof].dropna()
    if len(past) < 30:
        return 1.0
    window = past.tail(cfg.history_window + 1)
    if len(window) < 30:
        return 1.0
    history = window.iloc[:-1]
    current = float(window.iloc[-1])
    mu = float(history.mean())
    sd = float(history.std(ddof=1))
    if sd < 1e-12:
        return 1.0
    z = (current - mu) / sd
    # Boltzmann: when r is anomalously high → synchronized → de-risk
    target = float(np.exp(-max(0.0, z) / cfg.boltzmann_kt))
    target = float(np.clip(target, cfg.scale_floor, cfg.scale_ceiling))
    if prev_scale is not None and cfg.blend > 0:
        target = cfg.blend * prev_scale + (1 - cfg.blend) * target
    return float(np.clip(target, cfg.scale_floor, cfg.scale_ceiling))
