"""RMT-cleaned signal-tilted min-variance allocator (Phase 38).

Drop-in alternative to `inverse_vol_allocation`: same signature, same
caps and cash rules, but the weighting step uses a Marchenko-Pastur
denoised covariance matrix and a Markowitz-style max(wᵀμ - γ/2 wᵀΣw)
solve instead of "rebalanced inverse vol".

Theory: under the Marchenko-Pastur law, eigenvalues of an N×N sample
covariance with q=N/T iid Gaussian noise are bounded above by
(1+√q)². Eigenvalues inside the M-P window are noise — projecting them
to the noise mean preserves the trace while denoising the inverse Σ⁻¹
that drives Markowitz weights. This is the Laloux-Cizeau-Bouchaud-Potters
1999 recipe. It is *parameter-free* — no hyperparameters tuned on
val/test data.

The risk-aversion γ defaults to 5.0 (standard textbook choice for
unit-scale signals); we do not search over it.
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
from afp.portfolio.rmt_covariance import (
    clean_covariance_rmt,
    signal_tilted_min_var_weights,
)


def _build_returns_panel(ids: list[str], asof: date,
                         vol_estimator: VolEstimator,
                         lookback_days: int = 252) -> np.ndarray | None:
    """Stack daily log-returns from VolEstimator's per-company series."""
    series_list: list[pd.Series] = []
    for icid in ids:
        s = vol_estimator._by_company.get(icid)
        if s is None:
            return None
        past = s.loc[s.index < pd.Timestamp(asof)].tail(lookback_days)
        if len(past) < max(40, lookback_days // 4):
            return None
        series_list.append(past)
    df = pd.concat(series_list, axis=1, keys=ids).dropna()
    if len(df) < 40:
        return None
    return df.to_numpy()


def rmt_min_var_allocation(candidates: pd.DataFrame, cfg: PortfolioConfig,
                           vol_estimator: VolEstimator | None = None,
                           as_of: date | None = None,
                           lookback_days: int = 252,
                           risk_aversion: float = 5.0) -> AllocationResult:
    """RMT-denoised signal-tilted min-variance allocation."""
    if candidates.empty:
        return AllocationResult(weights=pd.DataFrame(columns=[
            "internal_company_id", "weight", "predicted_signal", "vol"]),
            cash_weight=1.0 if cfg.allow_cash else 0.0,
            n_candidates=0)

    ids = candidates["internal_company_id"].tolist()
    returns_panel = None
    if vol_estimator is not None and as_of is not None:
        returns_panel = _build_returns_panel(ids, as_of, vol_estimator, lookback_days)

    if returns_panel is None or returns_panel.shape[1] < 2:
        # Fall back to inverse-vol when we cannot build a cov matrix.
        # (Most likely an early backtest day with too little price history.)
        from afp.portfolio.allocation import inverse_vol_allocation
        return inverse_vol_allocation(candidates, cfg)

    cov = clean_covariance_rmt(returns_panel, keep_market_mode=True)
    signal = candidates["positive_signal"].to_numpy() ** cfg.signal_power
    if signal.sum() <= 0:
        signal = np.ones(len(signal))
    signal = signal / signal.sum()

    weights = signal_tilted_min_var_weights(
        cov, signal, risk_aversion=risk_aversion, long_only=True,
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


def build_rmt_target_portfolio(
    as_of: date,
    predictions_with_context: pd.DataFrame,
    vol_estimator: VolEstimator,
    cfg: PortfolioConfig,
    k: float = 2.5,
    sector_map: dict[str, str] | None = None,
    lookback_days: int = 252,
    risk_aversion: float = 5.0,
) -> AllocationResult:
    """End-to-end RMT pipeline (mirrors `build_target_portfolio`)."""
    from afp.portfolio.candidate_selection import active_predictions
    from afp.portfolio.signal_restore import add_restored_returns

    active = active_predictions(predictions_with_context, as_of)
    if active.empty:
        return rmt_min_var_allocation(pd.DataFrame(), cfg)

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
    return rmt_min_var_allocation(candidates, cfg,
                                  vol_estimator=vol_estimator, as_of=as_of,
                                  lookback_days=lookback_days,
                                  risk_aversion=risk_aversion)
