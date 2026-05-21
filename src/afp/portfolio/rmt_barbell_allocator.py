"""RMT-weighted barbell allocator (Phase 54).

Combines Phase 38 (RMT min-var) with Phase 53 (Taleb barbell):

- SAFE side (80%): RMT signal-tilted min-variance on ALL candidates.
- CONCENTRATED side (20%): RMT signal-tilted min-variance on the
  top-5 candidates by raw signal.

Both inner allocations are well-motivated; the outer 80/20 split is
the same Taleb barbell ratio used in Phase 53. All parameters are
pre-declared from theory or from earlier-phase defaults; no tuning.
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
from afp.portfolio.rmt_allocator import _build_returns_panel
from afp.portfolio.rmt_covariance import clean_covariance_rmt, signal_tilted_min_var_weights


def _rmt_weights_on_subset(candidates: pd.DataFrame, idx: np.ndarray,
                           vol_estimator: VolEstimator, as_of: date,
                           lookback_days: int = 252,
                           risk_aversion: float = 5.0) -> np.ndarray:
    """RMT signal-tilted min-var weights on a subset of candidates (indices in `idx`)."""
    sub = candidates.iloc[idx]
    ids = sub["internal_company_id"].tolist()
    returns_panel = _build_returns_panel(ids, as_of, vol_estimator, lookback_days)
    if returns_panel is None or returns_panel.shape[1] < 2:
        # Fallback: equal weight
        return np.full(len(idx), 1.0 / max(1, len(idx)))
    cov = clean_covariance_rmt(returns_panel)
    signal = sub["positive_signal"].to_numpy().astype(float) ** 1.0
    if signal.sum() <= 0:
        signal = np.ones(len(signal))
    signal = signal / signal.sum()
    w = signal_tilted_min_var_weights(cov, signal, risk_aversion=risk_aversion,
                                       long_only=True, max_weight=1.0)
    if w.sum() > 0:
        w = w / w.sum()
    return w


def rmt_barbell_allocation(candidates: pd.DataFrame, cfg: PortfolioConfig,
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

    n = len(candidates)
    if vol_estimator is None or as_of is None:
        from afp.portfolio.barbell_allocator import barbell_allocation
        return barbell_allocation(candidates, cfg,
                                  safe_fraction=safe_fraction,
                                  concentrated_top_k=concentrated_top_k)

    all_idx = np.arange(n)
    sorted_idx = candidates["positive_signal"].argsort()[::-1].to_numpy()
    top_idx = sorted_idx[:min(concentrated_top_k, n)]

    # Safe slice: RMT on all candidates × safe_fraction
    w_safe_raw = _rmt_weights_on_subset(candidates, all_idx, vol_estimator,
                                         as_of, lookback_days, risk_aversion)
    w_safe = np.zeros(n)
    w_safe[all_idx] = w_safe_raw * safe_fraction

    # Concentrated slice: RMT on top-K × (1 - safe_fraction)
    w_conc_raw = _rmt_weights_on_subset(candidates, top_idx, vol_estimator,
                                         as_of, lookback_days, risk_aversion)
    w_conc = np.zeros(n)
    w_conc[top_idx] = w_conc_raw * (1.0 - safe_fraction)

    weights = w_safe + w_conc

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


def build_rmt_barbell_target_portfolio(
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
    from afp.portfolio.candidate_selection import active_predictions
    from afp.portfolio.signal_restore import add_restored_returns

    active = active_predictions(predictions_with_context, as_of)
    if active.empty:
        return rmt_barbell_allocation(pd.DataFrame(), cfg)

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
    return rmt_barbell_allocation(candidates, cfg, vol_estimator=vol_estimator,
                                  as_of=as_of, safe_fraction=safe_fraction,
                                  concentrated_top_k=concentrated_top_k,
                                  risk_aversion=risk_aversion)
