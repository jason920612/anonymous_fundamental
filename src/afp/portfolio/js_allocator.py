"""James-Stein-shrunken RMT min-var allocator (Phase 48).

A signal-pre-processing wrapper that applies James-Stein shrinkage
to candidate signals before invoking the Phase 38 RMT min-var
allocator. The grand-mean component of the signal is preserved; the
deviations from it are shrunk by a theory-determined factor.

Parameter-free.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from afp.portfolio.allocation import AllocationResult, PortfolioConfig
from afp.portfolio.js_shrinkage import js_shrink
from afp.portfolio.risk_models import VolEstimator
from afp.portfolio.rmt_allocator import rmt_min_var_allocation


def _shrink_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return a new candidates DataFrame with JS-shrunken positive_signal."""
    df = candidates.copy()
    sig = df["positive_signal"].to_numpy().astype(float)
    df["positive_signal"] = js_shrink(sig)
    return df


def js_rmt_allocation(candidates: pd.DataFrame, cfg: PortfolioConfig,
                     vol_estimator: VolEstimator | None = None,
                     as_of: date | None = None,
                     lookback_days: int = 252,
                     risk_aversion: float = 5.0) -> AllocationResult:
    if candidates.empty:
        return AllocationResult(weights=pd.DataFrame(columns=[
            "internal_company_id", "weight", "predicted_signal", "vol"]),
            cash_weight=1.0 if cfg.allow_cash else 0.0,
            n_candidates=0)
    shrunk = _shrink_candidates(candidates)
    return rmt_min_var_allocation(shrunk, cfg, vol_estimator=vol_estimator,
                                  as_of=as_of, lookback_days=lookback_days,
                                  risk_aversion=risk_aversion)


def build_js_rmt_target_portfolio(
    as_of: date,
    predictions_with_context: pd.DataFrame,
    vol_estimator: VolEstimator,
    cfg: PortfolioConfig,
    k: float = 2.5,
    sector_map: dict[str, str] | None = None,
    lookback_days: int = 252,
    risk_aversion: float = 5.0,
) -> AllocationResult:
    from afp.portfolio.allocation import select_candidates
    from afp.portfolio.candidate_selection import active_predictions
    from afp.portfolio.signal_restore import add_restored_returns

    active = active_predictions(predictions_with_context, as_of)
    if active.empty:
        return js_rmt_allocation(pd.DataFrame(), cfg)

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
    return js_rmt_allocation(candidates, cfg, vol_estimator=vol_estimator,
                             as_of=as_of, lookback_days=lookback_days,
                             risk_aversion=risk_aversion)
