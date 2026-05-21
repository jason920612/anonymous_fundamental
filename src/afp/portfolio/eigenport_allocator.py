"""Eigenportfolio market-mode signal subtraction (Phase 47).

Theory:

The dominant eigenvector v₁ of the RMT-cleaned covariance matrix
represents the "market mode" — a direction in stock-space that all
assets load on with the same sign. A signal vector μ generally has
a component along v₁ (call it `μ · v̂₁`) and an orthogonal residual
component μ_⊥ = μ − (μ · v̂₁) v̂₁.

The component **along v₁** is "market-correlated alpha" — names
whose returns move with the market direction. In drawdowns these
names all decline together, so concentrating bets in this direction
produces large MDD.

The component **orthogonal to v₁** is "cross-sectional alpha" — the
part of the signal that distinguishes between assets relative to the
market mode. Concentrating bets here gives portfolio returns that are
mostly idiosyncratic.

For a long-only portfolio we cannot zero out v₁ exposure entirely
(weights are non-negative; a positive-loading v₁ means *any* long
portfolio has positive v₁ exposure). What we *can* do is feed only
the orthogonal component μ_⊥ into the min-variance solve, then let
the long-only / cap constraints handle the rest.

Result: the *tilt* of weights away from naive long-only is driven
purely by cross-sectional alpha. Hence: less concentration along
the market direction, better behaviour in correlated drawdowns.

This is parameter-free.
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
from afp.portfolio.rmt_covariance import clean_covariance_rmt, signal_tilted_min_var_weights


def project_out_eigenmode(signal: np.ndarray, eigvec: np.ndarray) -> np.ndarray:
    """Return the signal orthogonal to a normalized eigenvector."""
    v = eigvec / max(np.linalg.norm(eigvec), 1e-12)
    proj = float(signal @ v) * v
    return signal - proj


def eigenport_min_var_allocation(candidates: pd.DataFrame, cfg: PortfolioConfig,
                                 vol_estimator: VolEstimator | None = None,
                                 as_of: date | None = None,
                                 lookback_days: int = 252,
                                 risk_aversion: float = 5.0) -> AllocationResult:
    if candidates.empty:
        return AllocationResult(weights=pd.DataFrame(columns=[
            "internal_company_id", "weight", "predicted_signal", "vol"]),
            cash_weight=1.0 if cfg.allow_cash else 0.0,
            n_candidates=0)

    ids = candidates["internal_company_id"].tolist()
    returns_panel = None
    if vol_estimator is not None and as_of is not None:
        from afp.portfolio.rmt_allocator import _build_returns_panel
        returns_panel = _build_returns_panel(ids, as_of, vol_estimator, lookback_days)
    if returns_panel is None or returns_panel.shape[1] < 2:
        from afp.portfolio.allocation import inverse_vol_allocation
        return inverse_vol_allocation(candidates, cfg)

    cov = clean_covariance_rmt(returns_panel)
    # Dominant eigenvector (largest eigenvalue) — the market mode.
    eigvals, eigvecs = np.linalg.eigh(cov)
    v1 = eigvecs[:, -1]
    if (v1 < 0).sum() > (v1 > 0).sum():
        v1 = -v1  # flip sign convention so the dominant loading is positive

    signal = candidates["positive_signal"].to_numpy().astype(float) ** cfg.signal_power
    if signal.sum() <= 0:
        signal = np.ones(len(signal))
    signal = signal / signal.sum()
    signal_orth = project_out_eigenmode(signal, v1)

    # Re-normalize as a probability-like vector for the QP. Keep magnitude,
    # use the raw orthogonal component (may include negatives).
    weights = signal_tilted_min_var_weights(
        cov, signal_orth, risk_aversion=risk_aversion, long_only=True,
        max_weight=cfg.max_single_stock_weight)

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


def build_eigenport_target_portfolio(
    as_of: date,
    predictions_with_context: pd.DataFrame,
    vol_estimator: VolEstimator,
    cfg: PortfolioConfig,
    k: float = 2.5,
    sector_map: dict[str, str] | None = None,
    lookback_days: int = 252,
    risk_aversion: float = 5.0,
) -> AllocationResult:
    from afp.portfolio.candidate_selection import active_predictions
    from afp.portfolio.signal_restore import add_restored_returns

    active = active_predictions(predictions_with_context, as_of)
    if active.empty:
        return eigenport_min_var_allocation(pd.DataFrame(), cfg)

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
    return eigenport_min_var_allocation(candidates, cfg, vol_estimator=vol_estimator,
                                        as_of=as_of, lookback_days=lookback_days,
                                        risk_aversion=risk_aversion)
