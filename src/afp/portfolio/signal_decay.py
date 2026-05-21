"""Time-decay signal weighting (Phase 63, first principles).

First-principles claim: the signal generated at `entry_date` carries
the most information about future returns *immediately after* the
filing. As time elapses, market participants incorporate the same
fundamental information, eroding the signal's predictive edge.

By the time we approach `exit_date` (next filing), the current
signal is fully stale and a fresh signal is imminent.

Mathematical form: exponential decay with characteristic time τ:

    signal_eff(t) = signal_0 · exp(-(t - entry_date) / τ)

τ = median historical filing cadence ≈ 63 trading days (quarterly).

This is purely a SIGNAL TRANSFORMATION at the candidate-selection
step; it does not change the allocator structure. Pre-declared τ
from theory (one trading quarter); not tuned on val/test data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass
class TimeDecayConfig:
    enabled: bool = False
    tau_days: float = 63.0           # one trading quarter
    decay_mode: str = "exponential"  # or "linear"
    min_floor: float = 0.10          # never below 10% of original
    max_age_days: int = 252          # hard cutoff


def apply_time_decay(predictions: pd.DataFrame, as_of: date,
                     cfg: TimeDecayConfig) -> pd.DataFrame:
    """Apply time-decay to `predicted_signal` based on age since entry_date.

    Operates causally — only uses entry_date and the current as_of.
    Adds an `age_days` column and a `decay_factor` column for
    transparency.
    """
    if not cfg.enabled or predictions.empty:
        df = predictions.copy()
        df["age_days"] = 0
        df["decay_factor"] = 1.0
        return df
    df = predictions.copy()
    asof_ts = pd.Timestamp(as_of)
    entry_ts = pd.to_datetime(df["entry_date"])
    age = (asof_ts - entry_ts).dt.days.clip(lower=0)
    age = age.clip(upper=cfg.max_age_days)
    if cfg.decay_mode == "linear":
        # Linear decay from 1 at age=0 to floor at age=tau.
        factor = np.clip(1.0 - (age / cfg.tau_days) * (1.0 - cfg.min_floor),
                          cfg.min_floor, 1.0)
    else:
        factor = np.exp(-age / cfg.tau_days)
        factor = np.clip(factor, cfg.min_floor, 1.0)
    df["age_days"] = age.to_numpy()
    df["decay_factor"] = factor.to_numpy()
    df["predicted_signal"] = df["predicted_signal"].to_numpy() * factor.to_numpy()
    return df
