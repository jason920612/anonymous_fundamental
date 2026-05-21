"""Antifragile barbell allocator (Phase 53, after Taleb 2012).

Theory: split capital between a SAFE side and a CONCENTRATED side.

- SAFE side: equal-weight across ALL active candidates. Maximally
  diversified — captures broad market alpha with low MDD.
- CONCENTRATED side: equal-weight across the TOP K names by signal.
  Captures the model's high-conviction picks.

The "barbell" — heavy on both ends, nothing in the middle — gives:

- Strict floor on MDD (the safe side cannot single-name blow up).
- Asymmetric upside (the concentrated side participates in winners).

Parameters pre-declared from theory:
- safe_fraction = 0.80
- concentrated_top_k = 5
- equal_weight inside each slice (no further tuning)

No tuning on val/test data.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from afp.portfolio.allocation import (
    AllocationResult,
    PortfolioConfig,
    _apply_sector_cap,
    _cap_weights,
    select_candidates,
)
from afp.portfolio.risk_models import VolEstimator


def barbell_allocation(candidates: pd.DataFrame, cfg: PortfolioConfig,
                       safe_fraction: float = 0.80,
                       concentrated_top_k: int = 5) -> AllocationResult:
    if candidates.empty:
        return AllocationResult(weights=pd.DataFrame(columns=[
            "internal_company_id", "weight", "predicted_signal", "vol"]),
            cash_weight=1.0 if cfg.allow_cash else 0.0,
            n_candidates=0)

    n = len(candidates)
    # SAFE side: equal weight on every candidate
    w_safe = np.full(n, safe_fraction / n)

    # CONCENTRATED side: equal weight on top-K by positive_signal
    k = min(concentrated_top_k, n)
    sorted_idx = candidates["positive_signal"].argsort()[::-1]
    top_idx = sorted_idx[:k].to_numpy()
    w_conc = np.zeros(n)
    w_conc[top_idx] = (1.0 - safe_fraction) / k

    weights = w_safe + w_conc

    # Apply caps + cash like other allocators
    weights = np.where(weights < cfg.min_position_weight, 0.0, weights)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    weights = _cap_weights(weights, cfg.max_single_stock_weight)

    if cfg.max_sector_weight is not None and "sector" in candidates.columns:
        sectors = candidates["sector"].fillna("UNK").to_numpy()
        weights = _apply_sector_cap(weights, sectors, cfg.max_sector_weight)
        weights = _cap_weights(weights, cfg.max_single_stock_weight)

    if weights.sum() > cfg.leverage:
        weights = weights / weights.sum() * cfg.leverage
    cash_weight = max(0.0, cfg.leverage - weights.sum())
    if not cfg.allow_cash and cash_weight > 1e-9 and weights.sum() > 0:
        weights = weights / weights.sum() * cfg.leverage
        weights = _cap_weights(weights, cfg.max_single_stock_weight)
        cash_weight = max(0.0, cfg.leverage - weights.sum())

    out = pd.DataFrame({
        "internal_company_id": candidates["internal_company_id"].to_numpy(),
        "weight": weights,
        "predicted_signal": candidates["predicted_signal"].to_numpy(),
        "vol": candidates["vol_estimate"].to_numpy(),
    })
    out = out[out["weight"] > 0].reset_index(drop=True)
    return AllocationResult(weights=out, cash_weight=float(cash_weight),
                            n_candidates=len(candidates))


def build_barbell_target_portfolio(
    as_of: date,
    predictions_with_context: pd.DataFrame,
    vol_estimator: VolEstimator,
    cfg: PortfolioConfig,
    k: float = 2.5,
    sector_map: dict[str, str] | None = None,
    safe_fraction: float = 0.80,
    concentrated_top_k: int = 5,
) -> AllocationResult:
    from afp.portfolio.candidate_selection import active_predictions
    from afp.portfolio.signal_restore import add_restored_returns

    active = active_predictions(predictions_with_context, as_of)
    if active.empty:
        return barbell_allocation(pd.DataFrame(), cfg)

    restored = add_restored_returns(active, k=k)
    restored = restored.sort_values("entry_date").drop_duplicates(
        subset=["internal_company_id"], keep="last")
    vols = restored["internal_company_id"].map(
        lambda icid: vol_estimator.vol_at(icid, as_of))
    restored = restored.copy()
    restored["vol_estimate"] = vols.to_numpy()
    if sector_map is not None:
        restored["sector"] = restored["internal_company_id"].map(sector_map).fillna("UNK")
    candidates = select_candidates(restored, cfg)
    if sector_map is not None and "sector" in restored.columns:
        candidates = candidates.merge(restored[["internal_company_id", "sector"]],
                                      on="internal_company_id", how="left")
    return barbell_allocation(candidates, cfg,
                              safe_fraction=safe_fraction,
                              concentrated_top_k=concentrated_top_k)
