"""Michaelis-Menten saturated RMT min-var allocator (Phase 59).

Pre-processes the positive_signal vector with M-M saturation, then
feeds it through the Phase 54 winner stack (RMT-barbell). Compose
orthogonally — the M-M transform is purely signal-side, the RMT
barbell handles weighting structure.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from afp.portfolio.allocation import AllocationResult, PortfolioConfig
from afp.portfolio.mm_saturation import michaelis_menten
from afp.portfolio.risk_models import VolEstimator
from afp.portfolio.rmt_barbell_allocator import rmt_barbell_allocation


def _saturate_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    df = candidates.copy()
    df["positive_signal"] = michaelis_menten(df["positive_signal"].to_numpy())
    return df


def mm_rmt_barbell_allocation(candidates: pd.DataFrame, cfg: PortfolioConfig,
                              vol_estimator: VolEstimator | None = None,
                              as_of: date | None = None,
                              lookback_days: int = 252,
                              safe_fraction: float = 0.80,
                              concentrated_top_k: int = 5,
                              risk_aversion: float = 5.0) -> AllocationResult:
    if candidates.empty:
        return AllocationResult(weights=pd.DataFrame(columns=[
            "internal_company_id", "weight", "predicted_signal", "vol"]),
            cash_weight=1.0 if cfg.allow_cash else 0.0,
            n_candidates=0)
    saturated = _saturate_candidates(candidates)
    return rmt_barbell_allocation(saturated, cfg, vol_estimator=vol_estimator,
                                  as_of=as_of, lookback_days=lookback_days,
                                  safe_fraction=safe_fraction,
                                  concentrated_top_k=concentrated_top_k,
                                  risk_aversion=risk_aversion)


def build_mm_rmt_barbell_target_portfolio(
    as_of: date,
    predictions_with_context: pd.DataFrame,
    vol_estimator: VolEstimator,
    cfg: PortfolioConfig,
    k: float = 2.5,
    sector_map: dict[str, str] | None = None,
    safe_fraction: float = 0.80,
    concentrated_top_k: int = 5,
    risk_aversion: float = 5.0,
) -> AllocationResult:
    from afp.portfolio.allocation import select_candidates
    from afp.portfolio.candidate_selection import active_predictions
    from afp.portfolio.signal_restore import add_restored_returns

    active = active_predictions(predictions_with_context, as_of)
    if active.empty:
        return mm_rmt_barbell_allocation(pd.DataFrame(), cfg)

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
    return mm_rmt_barbell_allocation(candidates, cfg, vol_estimator=vol_estimator,
                                     as_of=as_of, safe_fraction=safe_fraction,
                                     concentrated_top_k=concentrated_top_k,
                                     risk_aversion=risk_aversion)
