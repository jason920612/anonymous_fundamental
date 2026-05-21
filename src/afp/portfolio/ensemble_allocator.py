"""Allocator-level ensemble (Phase 51).

Hypothesis: two allocators with fundamentally different failure modes
should, when averaged, produce a weight vector that is more robust
than either alone.

The two allocators we ensemble:

- Inverse-vol (baseline): weight ∝ signal / σ_i. Failure mode: ignores
  correlation structure, over-allocates to correlated names.

- RMT min-var (Phase 38): weight ∝ Σ_clean⁻¹ · signal. Failure mode:
  sensitive to small eigenvalues — even after M-P cleaning, very small
  cov eigenvalues amplify in the inverse and can produce extreme tilts.

Averaging: `w_ens = α · w_inv_vol + (1 - α) · w_rmt`.

α=0.5 is the theoretical mid-point (no preference between allocators).
Not tuned on val/test data. Applied AFTER each allocator's own caps;
re-cap the ensemble result to honor max_single_stock_weight.
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
    inverse_vol_allocation,
    select_candidates,
)
from afp.portfolio.risk_models import VolEstimator
from afp.portfolio.rmt_allocator import rmt_min_var_allocation


def ensemble_allocation(candidates: pd.DataFrame, cfg: PortfolioConfig,
                       vol_estimator: VolEstimator | None = None,
                       as_of: date | None = None,
                       lookback_days: int = 252,
                       alpha: float = 0.5) -> AllocationResult:
    """Mean of inverse-vol and RMT min-var weight vectors."""
    if candidates.empty:
        return AllocationResult(weights=pd.DataFrame(columns=[
            "internal_company_id", "weight", "predicted_signal", "vol"]),
            cash_weight=1.0 if cfg.allow_cash else 0.0,
            n_candidates=0)

    inv_vol = inverse_vol_allocation(candidates, cfg)
    rmt = rmt_min_var_allocation(candidates, cfg, vol_estimator=vol_estimator,
                                  as_of=as_of, lookback_days=lookback_days)
    # Build a merged DataFrame on internal_company_id.
    iv = inv_vol.weights.set_index("internal_company_id")["weight"]
    rw = rmt.weights.set_index("internal_company_id")["weight"]
    ids = candidates["internal_company_id"].to_numpy()
    iv_aligned = iv.reindex(ids).fillna(0.0).to_numpy()
    rmt_aligned = rw.reindex(ids).fillna(0.0).to_numpy()
    weights = alpha * iv_aligned + (1 - alpha) * rmt_aligned

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


def build_ensemble_target_portfolio(
    as_of: date,
    predictions_with_context: pd.DataFrame,
    vol_estimator: VolEstimator,
    cfg: PortfolioConfig,
    k: float = 2.5,
    sector_map: dict[str, str] | None = None,
    lookback_days: int = 252,
    alpha: float = 0.5,
) -> AllocationResult:
    from afp.portfolio.candidate_selection import active_predictions
    from afp.portfolio.signal_restore import add_restored_returns

    active = active_predictions(predictions_with_context, as_of)
    if active.empty:
        return ensemble_allocation(pd.DataFrame(), cfg)

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
    return ensemble_allocation(candidates, cfg, vol_estimator=vol_estimator,
                               as_of=as_of, lookback_days=lookback_days, alpha=alpha)
