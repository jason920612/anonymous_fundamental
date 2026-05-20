"""Phase 24: distribution-aware long-only allocator.

When predictions include `probability_positive` and `predicted_signal_std`
(diffusion output), this allocator:

  1. Filters candidates by `probability_positive >= min_prob` instead of
     `predicted_signal > min_positive_signal` alone.
  2. Sizes each position proportional to a *confidence-weighted* signal:
        confidence = max(probability_positive - 0.5, 0)
        position_score = confidence / max(std, std_floor)
     This rewards high-conviction picks (prob far from 50%) and downweights
     uncertain ones.
  3. Falls back to the deterministic `inverse_vol_allocation` path when the
     distribution columns are absent — so non-diffusion models still work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from afp.portfolio.allocation import (
    AllocationResult,
    PortfolioConfig,
    _apply_sector_cap,
    _cap_weights,
    inverse_vol_allocation,
)


@dataclass
class DistributionAllocatorConfig:
    min_probability_positive: float = 0.55
    std_floor: float = 0.05
    confidence_power: float = 1.0
    blend_with_signal: float = 0.0     # 0 → pure confidence; 1 → also weight by predicted mean


def _has_distribution_cols(candidates: pd.DataFrame) -> bool:
    return ("probability_positive" in candidates.columns
            and "predicted_signal_std" in candidates.columns)


def select_distribution_candidates(prediction_rows: pd.DataFrame,
                                   port_cfg: PortfolioConfig,
                                   dist_cfg: DistributionAllocatorConfig
                                   ) -> pd.DataFrame:
    """Filter by probability_positive + ensure vol_estimate present."""
    df = prediction_rows.dropna(subset=["predicted_signal", "vol_estimate",
                                        "probability_positive", "predicted_signal_std"]).copy()
    df = df[df["probability_positive"] >= dist_cfg.min_probability_positive]
    if df.empty:
        return df
    df["positive_signal"] = df["predicted_signal"].clip(lower=0.0)
    df["confidence"] = (df["probability_positive"] - 0.5).clip(lower=0.0)
    df["sort_key"] = df["confidence"] / df["predicted_signal_std"].clip(lower=dist_cfg.std_floor)
    df = df.sort_values("sort_key", ascending=False).head(port_cfg.max_positions)
    return df.reset_index(drop=True)


def distribution_aware_allocation(candidates: pd.DataFrame,
                                  port_cfg: PortfolioConfig,
                                  dist_cfg: DistributionAllocatorConfig
                                  ) -> AllocationResult:
    """Inverse-vol weighting with risk budget driven by confidence/std."""
    if candidates.empty:
        return AllocationResult(
            weights=pd.DataFrame(columns=["internal_company_id", "weight",
                                          "predicted_signal", "vol"]),
            cash_weight=1.0 if port_cfg.allow_cash else 0.0,
            n_candidates=0,
        )

    confidence = candidates["confidence"].to_numpy()
    std = candidates["predicted_signal_std"].clip(lower=dist_cfg.std_floor).to_numpy()
    if dist_cfg.blend_with_signal > 0:
        signal = candidates["predicted_signal"].clip(lower=0.0).to_numpy()
        rb_raw = (confidence ** dist_cfg.confidence_power) * np.power(
            np.maximum(signal, 1e-9), dist_cfg.blend_with_signal
        ) / std
    else:
        rb_raw = (confidence ** dist_cfg.confidence_power) / std

    if rb_raw.sum() <= 0:
        weights = np.full(len(candidates), 1.0 / len(candidates))
    else:
        risk_budget = rb_raw / rb_raw.sum()
        raw_w = risk_budget / candidates["vol_estimate"].to_numpy()
        if raw_w.sum() <= 0:
            weights = np.full(len(candidates), 1.0 / len(candidates))
        else:
            weights = raw_w / raw_w.sum()

    # Same tail handling as inverse_vol_allocation
    weights = np.where(weights < port_cfg.min_position_weight, 0.0, weights)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    weights = _cap_weights(weights, port_cfg.max_single_stock_weight)
    if port_cfg.max_sector_weight is not None and "sector" in candidates.columns:
        sectors = candidates["sector"].fillna("UNK").to_numpy()
        weights = _apply_sector_cap(weights, sectors, port_cfg.max_sector_weight)
        weights = _cap_weights(weights, port_cfg.max_single_stock_weight)
    weight_sum = weights.sum()
    if weight_sum > port_cfg.leverage:
        weights = weights / weight_sum * port_cfg.leverage

    cash_weight = max(0.0, port_cfg.leverage - weights.sum())
    if not port_cfg.allow_cash and cash_weight > 1e-9 and weights.sum() > 0:
        weights = weights / weights.sum() * port_cfg.leverage
        weights = _cap_weights(weights, port_cfg.max_single_stock_weight)
        cash_weight = max(0.0, port_cfg.leverage - weights.sum())

    out = pd.DataFrame({
        "internal_company_id": candidates["internal_company_id"].to_numpy(),
        "weight": weights,
        "predicted_signal": candidates["predicted_signal"].to_numpy(),
        "vol": candidates["vol_estimate"].to_numpy(),
        "probability_positive": candidates["probability_positive"].to_numpy(),
        "confidence": confidence,
        "predicted_signal_std": std,
    })
    out = out[out["weight"] > 0].reset_index(drop=True)
    return AllocationResult(weights=out, cash_weight=float(cash_weight),
                            n_candidates=len(candidates))


def build_distribution_portfolio(as_of, predictions_with_context,
                                 vol_estimator, port_cfg, dist_cfg,
                                 k: float = 2.5,
                                 sector_map: dict | None = None):
    """Drop-in replacement for `build_target_portfolio` that uses the distribution allocator."""
    from afp.portfolio.candidate_selection import active_predictions
    from afp.portfolio.signal_restore import add_restored_returns

    active = active_predictions(predictions_with_context, as_of)
    if active.empty or not _has_distribution_cols(active):
        from afp.portfolio.candidate_selection import build_target_portfolio
        return build_target_portfolio(as_of, predictions_with_context, vol_estimator,
                                       port_cfg, k=k, sector_map=sector_map)

    restored = add_restored_returns(active, k=k)
    restored = restored.sort_values("entry_date").drop_duplicates(
        subset=["internal_company_id"], keep="last")
    restored["vol_estimate"] = restored["internal_company_id"].map(
        lambda icid: vol_estimator.vol_at(icid, as_of)).to_numpy()
    if sector_map is not None:
        restored["sector"] = restored["internal_company_id"].map(sector_map).fillna("UNK")

    candidates = select_distribution_candidates(restored, port_cfg, dist_cfg)
    return distribution_aware_allocation(candidates, port_cfg, dist_cfg)
