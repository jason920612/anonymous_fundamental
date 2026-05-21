"""Phase 63: time-decay signal tests."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from afp.portfolio.signal_decay import TimeDecayConfig, apply_time_decay


def test_disabled_returns_unchanged():
    df = pd.DataFrame({
        "predicted_signal": [0.5, 0.3, 0.7],
        "entry_date": [pd.Timestamp("2020-01-01")] * 3,
    })
    cfg = TimeDecayConfig(enabled=False)
    out = apply_time_decay(df, date(2020, 1, 30), cfg)
    assert np.allclose(out["predicted_signal"].to_numpy(),
                       df["predicted_signal"].to_numpy())


def test_zero_age_no_decay():
    df = pd.DataFrame({
        "predicted_signal": [0.5],
        "entry_date": [pd.Timestamp("2020-01-15")],
    })
    cfg = TimeDecayConfig(enabled=True, tau_days=63.0)
    out = apply_time_decay(df, date(2020, 1, 15), cfg)
    # age=0 → factor=1
    assert abs(out["predicted_signal"].iloc[0] - 0.5) < 1e-9


def test_one_tau_age_decays_to_e_inverse():
    df = pd.DataFrame({
        "predicted_signal": [1.0],
        "entry_date": [pd.Timestamp("2020-01-01")],
    })
    cfg = TimeDecayConfig(enabled=True, tau_days=63.0, min_floor=0.01)
    # 63 days later → exp(-1) ≈ 0.368
    out = apply_time_decay(df, date(2020, 3, 4), cfg)
    factor = out["decay_factor"].iloc[0]
    assert 0.35 < factor < 0.40


def test_age_clipped_to_max():
    df = pd.DataFrame({
        "predicted_signal": [1.0],
        "entry_date": [pd.Timestamp("2020-01-01")],
    })
    cfg = TimeDecayConfig(enabled=True, tau_days=63.0, max_age_days=200,
                         min_floor=0.05)
    # 1000 days later → still capped at age=200 → factor=exp(-200/63) ≈ 0.042 but min floor=0.05
    out = apply_time_decay(df, date(2022, 10, 1), cfg)
    factor = out["decay_factor"].iloc[0]
    assert factor >= 0.05


def test_linear_decay():
    df = pd.DataFrame({
        "predicted_signal": [1.0],
        "entry_date": [pd.Timestamp("2020-01-01")],
    })
    cfg = TimeDecayConfig(enabled=True, tau_days=63.0, decay_mode="linear",
                         min_floor=0.1)
    # 63 days later → linear at tau → factor = min_floor = 0.1
    out = apply_time_decay(df, date(2020, 3, 4), cfg)
    factor = out["decay_factor"].iloc[0]
    assert abs(factor - 0.1) < 0.02
