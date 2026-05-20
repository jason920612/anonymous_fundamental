"""Phase 27: blended equal-risk-active + model-tilt allocator.

Hypothesis H1: the model's per-event signal is weak, but its *direction* is
slightly more correct than random. Equal-risk-active over the whole active
universe already gets Sharpe 1.44; we want to keep that diversification and
add a small multiplicative tilt toward high-confidence model picks.

Allocation:
    base_weight_i = (1/vol_i) / Σ(1/vol_j)              # equal-risk-active
    tilt_i        = (1 + α * (probability_positive_i - 0.5))  if dist cols available
                  | (1 + α * sign(predicted_signal_i)) * confidence    else fallback
    weight_i      = base_weight_i * tilt_i / Σ_j base_j * tilt_j
    apply caps, leverage

α ∈ [0, 2] roughly. α=0 → pure equal-risk-active. α=1 → strong tilt.

This is *not* a long-only filter — every active name is held. The tilt
multiplies weight up or down, but every name still gets at least
(1 - α*0.5)·base_weight, which is positive for α < 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from afp.portfolio.allocation import (
    AllocationResult,
    PortfolioConfig,
    _apply_sector_cap,
    _cap_weights,
)


@dataclass
class BlendedAllocatorConfig:
    tilt_alpha: float = 0.50
    use_probability_positive: bool = True
    min_active_universe: int = 50
    require_positive_signal: bool = False   # if True, weight drops to 0 when sig<=0


def _has_probability(predictions: pd.DataFrame) -> bool:
    return "probability_positive" in predictions.columns


def build_blended_portfolio(
    as_of: date,
    predictions_with_context: pd.DataFrame,
    vol_estimator,
    port_cfg: PortfolioConfig,
    blend_cfg: BlendedAllocatorConfig,
    k: float = 2.5,
    sector_map: dict | None = None,
):
    from afp.portfolio.candidate_selection import active_predictions
    from afp.portfolio.signal_restore import add_restored_returns

    active = active_predictions(predictions_with_context, as_of)
    if active.empty:
        return AllocationResult(
            weights=pd.DataFrame(columns=["internal_company_id", "weight",
                                          "predicted_signal", "vol"]),
            cash_weight=1.0 if port_cfg.allow_cash else 0.0,
            n_candidates=0,
        )

    df = add_restored_returns(active, k=k)
    df = df.sort_values("entry_date").drop_duplicates(subset=["internal_company_id"], keep="last")
    df["vol_estimate"] = df["internal_company_id"].map(lambda icid: vol_estimator.vol_at(icid, as_of))
    df = df.dropna(subset=["vol_estimate"])
    if df.empty or len(df) < blend_cfg.min_active_universe:
        # fall back to inverse_vol of whatever there is
        from afp.portfolio.allocation import inverse_vol_allocation, select_candidates
        cands = select_candidates(df, port_cfg)
        return inverse_vol_allocation(cands, port_cfg)
    if sector_map is not None:
        df["sector"] = df["internal_company_id"].map(sector_map).fillna("UNK")

    vol = df["vol_estimate"].to_numpy()
    base = (1.0 / vol) / np.sum(1.0 / vol)

    if blend_cfg.use_probability_positive and _has_probability(df):
        p = df["probability_positive"].fillna(0.5).to_numpy()
        tilt = 1.0 + blend_cfg.tilt_alpha * (p - 0.5)
    else:
        sig = df["predicted_signal"].fillna(0.0).to_numpy()
        tilt = 1.0 + blend_cfg.tilt_alpha * np.tanh(2.0 * sig)
    if blend_cfg.require_positive_signal:
        sig = df["predicted_signal"].fillna(0.0).to_numpy()
        tilt = np.where(sig > 0, tilt, 0.0)
    tilt = np.clip(tilt, 0.0, None)

    raw = base * tilt
    if raw.sum() <= 0:
        weights = np.full(len(df), 1.0 / len(df))
    else:
        weights = raw / raw.sum()

    weights = _cap_weights(weights, port_cfg.max_single_stock_weight)
    if port_cfg.max_sector_weight is not None and "sector" in df.columns:
        sectors = df["sector"].fillna("UNK").to_numpy()
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
        "internal_company_id": df["internal_company_id"].to_numpy(),
        "weight": weights,
        "predicted_signal": df["predicted_signal"].to_numpy(),
        "vol": vol,
    })
    out = out[out["weight"] > 0].reset_index(drop=True)
    return AllocationResult(weights=out, cash_weight=float(cash_weight),
                            n_candidates=len(df))
