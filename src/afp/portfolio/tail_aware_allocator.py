"""Tail-risk aware allocator (Phase 43 candidate).

Stock returns are NOT Gaussian — they have power-law tails (Mandelbrot
1963; Gopikrishnan-Plerou-Stanley 1998). Estimating risk via Gaussian
std systematically *under*-prices the contribution of extreme moves
to portfolio drawdown.

This allocator replaces Gaussian volatility with a **t-distribution
scale parameter** that down-weights names with heavy tails, then
applies HRP-style risk-parity weighting. Conceptually similar to a
CVaR / Expected Shortfall approach but cheaper to compute.

The single parameter is the t-distribution's degrees of freedom ν.
For US equities, empirical estimates from the econophysics literature
put ν in the range [3, 4] — heavy enough that variance is finite but
fourth moment diverges. We use ν=4 as a default consistent with the
Mandelbrot-Stanley estimates, and do not tune it on val/test data.

Tail-aware scale (Student-t MLE approximation):

    σ_t = scale of the t-distribution fitted by method-of-moments to
          the return series.

For a t-distribution with ν > 2,

    var = σ² · ν / (ν - 2)   ⇒   σ_t² = var · (ν - 2) / ν.

That is, σ_t < σ_gauss whenever ν > 2, but the *ratio* depends only
on ν. To actually penalize heavy tails, we instead use a **realized
ES** estimator:

    ES_α = -E[ r | r ≤ -VaR_α ]                       (left tail)

and assign per-asset risk = ES (annualized) instead of std. Names with
fatter left tails get higher risk, lower weight.
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


def expected_shortfall(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Empirical expected shortfall (CVaR) at confidence level alpha.

    ES_α = -mean of the bottom α-fraction of returns. Always non-negative.
    """
    n = len(returns)
    if n < int(1.0 / alpha):
        return float(np.std(returns, ddof=1)) if n > 1 else 0.0
    cutoff = max(1, int(alpha * n))
    sorted_r = np.sort(returns)
    tail = sorted_r[:cutoff]
    return float(-tail.mean())


def tail_risk_weights(es_values: np.ndarray, signal: np.ndarray,
                      power: float = 1.0) -> np.ndarray:
    """Inverse-ES weighting: w_i ∝ (signal_i / ES_i)^power, simplex projection."""
    es = np.where(es_values > 1e-9, es_values, 1e-9)
    raw = np.maximum(signal, 0.0) / es
    if power != 1.0:
        raw = np.sign(raw) * np.abs(raw) ** power
    if raw.sum() <= 0:
        return np.full(len(es), 1.0 / len(es))
    return raw / raw.sum()


def tail_aware_allocation(candidates: pd.DataFrame, cfg: PortfolioConfig,
                          vol_estimator: VolEstimator | None = None,
                          as_of: date | None = None,
                          lookback_days: int = 252,
                          es_alpha: float = 0.05) -> AllocationResult:
    """Long-only allocation with Expected-Shortfall-based risk weighting."""
    if candidates.empty:
        return AllocationResult(weights=pd.DataFrame(columns=[
            "internal_company_id", "weight", "predicted_signal", "vol"]),
            cash_weight=1.0 if cfg.allow_cash else 0.0,
            n_candidates=0)

    ids = candidates["internal_company_id"].tolist()
    es_values = []
    if vol_estimator is not None and as_of is not None:
        for icid in ids:
            s = vol_estimator._by_company.get(icid)
            if s is None:
                es_values.append(None)
                continue
            past = s.loc[s.index < pd.Timestamp(as_of)].tail(lookback_days)
            if len(past) < 40:
                es_values.append(None)
                continue
            es_values.append(expected_shortfall(past.to_numpy(), alpha=es_alpha))

    if any(v is None for v in es_values):
        # Fallback: any missing means we cannot use ES; revert to inverse-vol.
        from afp.portfolio.allocation import inverse_vol_allocation
        return inverse_vol_allocation(candidates, cfg)

    es_arr = np.array(es_values, dtype=float)
    signal = candidates["positive_signal"].to_numpy().astype(float) ** cfg.signal_power
    weights = tail_risk_weights(es_arr, signal)

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


def build_tail_aware_target_portfolio(
    as_of: date,
    predictions_with_context: pd.DataFrame,
    vol_estimator: VolEstimator,
    cfg: PortfolioConfig,
    k: float = 2.5,
    sector_map: dict[str, str] | None = None,
    lookback_days: int = 252,
    es_alpha: float = 0.05,
) -> AllocationResult:
    from afp.portfolio.candidate_selection import active_predictions
    from afp.portfolio.signal_restore import add_restored_returns

    active = active_predictions(predictions_with_context, as_of)
    if active.empty:
        return tail_aware_allocation(pd.DataFrame(), cfg)

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
    return tail_aware_allocation(candidates, cfg, vol_estimator=vol_estimator, as_of=as_of,
                                 lookback_days=lookback_days, es_alpha=es_alpha)
