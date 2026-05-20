"""Glue layer: take prediction rows + risk estimator → allocation-ready frame."""

from __future__ import annotations

from datetime import date

import pandas as pd

from afp.portfolio.allocation import (
    AllocationResult,
    PortfolioConfig,
    inverse_vol_allocation,
    select_candidates,
)
from afp.portfolio.risk_models import VolEstimator
from afp.portfolio.signal_restore import add_restored_returns


def active_predictions(predictions: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Predictions whose signal-validity window covers `as_of`."""
    asof = pd.Timestamp(as_of)
    e = pd.to_datetime(predictions["entry_date"])
    x = pd.to_datetime(predictions["exit_date"])
    return predictions[(e <= asof) & (asof <= x)]


def build_target_portfolio(
    as_of: date,
    predictions_with_context: pd.DataFrame,
    vol_estimator: VolEstimator,
    cfg: PortfolioConfig,
    k: float = 2.5,
    sector_map: dict[str, str] | None = None,
) -> AllocationResult:
    """End-to-end pipeline: restore signals → attach vol → select → allocate.

    `sector_map` (Phase 16): optional mapping `internal_company_id -> sector` so the
    allocator can apply `cfg.max_sector_weight`. If None, sector caps are skipped.
    """
    active = active_predictions(predictions_with_context, as_of)
    if active.empty:
        return inverse_vol_allocation(pd.DataFrame(), cfg)

    restored = add_restored_returns(active, k=k)
    # If multiple active predictions per company exist (overlap), keep latest entry_date
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
    return inverse_vol_allocation(candidates, cfg)
