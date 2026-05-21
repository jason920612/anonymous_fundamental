"""Adaptive-γ RMT min-var allocator (Phase 49).

At each rebalance, compute the Shannon entropy of the candidate
signal vector and map it (via a pre-declared function from theory)
to a risk-aversion γ ∈ [2, 10]. Then invoke the Phase 38 RMT
allocator with this γ.

No knob tuned on val/test. The mapping is closed-form.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from afp.portfolio.adaptive_gamma import adaptive_gamma
from afp.portfolio.allocation import AllocationResult, PortfolioConfig
from afp.portfolio.risk_models import VolEstimator
from afp.portfolio.rmt_allocator import rmt_min_var_allocation


def adaptive_rmt_allocation(candidates: pd.DataFrame, cfg: PortfolioConfig,
                            vol_estimator: VolEstimator | None = None,
                            as_of: date | None = None,
                            lookback_days: int = 252,
                            gamma_min: float = 2.0,
                            gamma_max: float = 10.0) -> AllocationResult:
    if candidates.empty:
        return AllocationResult(weights=pd.DataFrame(columns=[
            "internal_company_id", "weight", "predicted_signal", "vol"]),
            cash_weight=1.0 if cfg.allow_cash else 0.0,
            n_candidates=0)
    sig = candidates["positive_signal"].to_numpy().astype(float)
    gamma = adaptive_gamma(sig, gamma_min=gamma_min, gamma_max=gamma_max)
    return rmt_min_var_allocation(candidates, cfg, vol_estimator=vol_estimator,
                                  as_of=as_of, lookback_days=lookback_days,
                                  risk_aversion=gamma)


def build_adaptive_rmt_target_portfolio(
    as_of: date,
    predictions_with_context: pd.DataFrame,
    vol_estimator: VolEstimator,
    cfg: PortfolioConfig,
    k: float = 2.5,
    sector_map: dict[str, str] | None = None,
    lookback_days: int = 252,
    gamma_min: float = 2.0,
    gamma_max: float = 10.0,
) -> AllocationResult:
    from afp.portfolio.allocation import select_candidates
    from afp.portfolio.candidate_selection import active_predictions
    from afp.portfolio.signal_restore import add_restored_returns

    active = active_predictions(predictions_with_context, as_of)
    if active.empty:
        return adaptive_rmt_allocation(pd.DataFrame(), cfg)

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
    return adaptive_rmt_allocation(candidates, cfg, vol_estimator=vol_estimator,
                                   as_of=as_of, lookback_days=lookback_days,
                                   gamma_min=gamma_min, gamma_max=gamma_max)
