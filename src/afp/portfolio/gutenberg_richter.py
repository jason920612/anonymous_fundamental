"""Gutenberg-Richter foreshock detector (Phase 60).

Geophysics analog: large earthquakes are often preceded by clusters
of small foreshocks. The Gutenberg-Richter power law

    log10(N(M ≥ m)) = a - b·m

means small events are exponentially more common than large ones.
A *deviation* from this distribution — specifically, an excess of
small-magnitude events in a recent window — is a known precursor
to larger events.

Applied to portfolio drawdowns:

- Define "small drawdown event" = single-day strategy return < -kσ
  (we use k=1 for clear empirical countability; theory-only).
- Count such events over a rolling 60-day window.
- The expected count under Gaussian iid is ≈ 0.16·N (≈15.9% one-tail
  z<-1).
- z-score the actual count against its own rolling history.
- If z > 2 (statistically anomalous foreshock cluster), reduce
  exposure exponentially via Boltzmann factor (kT=2).

Pre-declared parameters from theory; no tuning on val/test data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GutenbergRichterConfig:
    enabled: bool = False
    event_sigma: float = 1.0       # daily ret < -kσ counts as a "foreshock"
    sigma_window: int = 60         # window for σ estimate
    count_window: int = 60         # window over which to count events
    history_window: int = 252      # rolling history for z-score of count
    boltzmann_kt: float = 2.0      # damping strength
    scale_floor: float = 0.30
    scale_ceiling: float = 1.0
    blend: float = 0.5


def _foreshock_count(daily_returns: pd.Series, cfg: GutenbergRichterConfig) -> int:
    """Number of single-day events with return below -kσ in the recent count window."""
    if len(daily_returns) < cfg.sigma_window + 5:
        return 0
    sigma_series = daily_returns.rolling(cfg.sigma_window, min_periods=10).std(ddof=1)
    sigma = float(sigma_series.iloc[-1])
    if not np.isfinite(sigma) or sigma <= 0:
        return 0
    threshold = -cfg.event_sigma * sigma
    recent = daily_returns.tail(cfg.count_window)
    return int((recent < threshold).sum())


def compute_gr_scale(daily_returns: pd.Series, cfg: GutenbergRichterConfig,
                     prev_scale: float | None = None) -> float:
    """Causal scale based on Gutenberg-Richter foreshock count z-score."""
    n_needed = cfg.history_window + cfg.count_window + 5
    if not cfg.enabled or len(daily_returns) < n_needed:
        return 1.0
    # Build a rolling series of foreshock counts (causal).
    rolling_counts = []
    for i in range(len(daily_returns) - cfg.history_window, len(daily_returns) + 1):
        sub = daily_returns.iloc[:i]
        rolling_counts.append(_foreshock_count(sub, cfg))
    counts = pd.Series(rolling_counts[:-1])  # exclude current = "now"
    current_count = rolling_counts[-1]
    mu = float(counts.mean())
    sd = float(counts.std(ddof=1))
    if sd < 1e-12:
        return 1.0
    z = (current_count - mu) / sd
    # Boltzmann damping on positive z only — anomalous foreshock cluster
    target = float(np.exp(-max(0.0, z) / cfg.boltzmann_kt))
    target = float(np.clip(target, cfg.scale_floor, cfg.scale_ceiling))
    if prev_scale is not None and cfg.blend > 0:
        target = cfg.blend * prev_scale + (1 - cfg.blend) * target
    return float(np.clip(target, cfg.scale_floor, cfg.scale_ceiling))
