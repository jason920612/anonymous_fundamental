"""tanh target transform + atanh restoration (RFC-04 §8-12).

Phase 13 added an optional `target_mode = cross_sectional_rank` so the model
can be trained against a same-quarter rank target (centered at 0, scaled to
[-1, 1]). The absolute-magnitude tanh target is retained as the default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TargetConfig:
    k: float = 2.5
    transform: str = "tanh"
    restore_method: str = "atanh"  # atanh | linear
    restore_clip_abs: float = 0.99
    target_mode: str = "tanh_scaled"           # tanh_scaled | cross_sectional_rank | sector_neutral_rank
    rank_period: str = "Q"                      # pandas .dt.to_period code for rank cohort


def apply(samples: pd.DataFrame, config: TargetConfig) -> pd.DataFrame:
    """Populate `target_normalized_signal` per `config.target_mode`.

    Modes:
      - `tanh_scaled` (default, RFC-04 §8): tanh(raw_log_return / (k * ex_ante_scale))
      - `cross_sectional_rank`: 2 * pct_rank_within(quarter) - 1; ex-ante scale
        is preserved for portfolio restoration but is not used in the target.

    Rows without `ex_ante_scale` are flagged `eligible_for_training = False`
    with `exclusion_reason = missing_ex_ante_scale` regardless of mode, since
    the portfolio module still needs the scale to size positions.
    """
    if samples.empty:
        return samples.copy()
    out = samples.copy()
    valid = out["ex_ante_scale"].notna() & (out["ex_ante_scale"] > 0)

    if config.target_mode == "cross_sectional_rank":
        out["target_normalized_signal"] = np.nan
        if valid.any():
            entry_period = pd.to_datetime(out.loc[valid, "entry_date"]).dt.to_period(config.rank_period)
            grp = out.loc[valid].assign(_entry_period=entry_period.values)
            # 2*pct_rank - 1 ∈ [-1, 1]; average rank handles ties
            grp["_rank"] = grp.groupby("_entry_period")["raw_log_return"].rank(method="average",
                                                                                 pct=True)
            out.loc[valid, "target_normalized_signal"] = (2.0 * grp["_rank"].to_numpy() - 1.0)
    elif config.target_mode == "sector_neutral_rank":
        out["target_normalized_signal"] = np.nan
        if "sector" not in out.columns:
            raise ValueError("sector_neutral_rank target requires a 'sector' column on samples")
        if valid.any():
            entry_period = pd.to_datetime(out.loc[valid, "entry_date"]).dt.to_period(config.rank_period)
            grp = out.loc[valid].assign(_entry_period=entry_period.astype(str).values,
                                        _sector=out.loc[valid, "sector"].fillna("UNK").values)
            grp["_rank"] = grp.groupby(["_entry_period", "_sector"])["raw_log_return"].rank(
                method="average", pct=True)
            out.loc[valid, "target_normalized_signal"] = (2.0 * grp["_rank"].to_numpy() - 1.0)
    else:
        k = float(config.k)
        out.loc[valid, "target_normalized_signal"] = np.tanh(
            out.loc[valid, "raw_log_return"] / (k * out.loc[valid, "ex_ante_scale"])
        )

    out.loc[~valid, "eligible_for_training"] = False
    existing = out["exclusion_reason"].fillna("")
    out.loc[~valid & (existing == ""), "exclusion_reason"] = "missing_ex_ante_scale"

    out["target_normalized_signal"] = out["target_normalized_signal"].clip(-1.0, 1.0)
    return out


def restore(predicted_signal: np.ndarray | float,
            ex_ante_scale: np.ndarray | float,
            config: TargetConfig) -> np.ndarray | float:
    """Restore predicted signal → expected log return. RFC-04 §10-12."""
    psig = np.asarray(predicted_signal, dtype=float)
    scale = np.asarray(ex_ante_scale, dtype=float)
    if config.restore_method == "linear":
        return psig * config.k * scale
    clipped = np.clip(psig, -config.restore_clip_abs, config.restore_clip_abs)
    return np.arctanh(clipped) * config.k * scale
