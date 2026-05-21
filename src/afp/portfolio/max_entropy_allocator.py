"""Maximum-entropy allocator (Phase 40).

Jaynes 1957 (max-entropy principle), Bera & Park 2008 (portfolio
diversification). Solve:

    maximise  H(w) = -Σ w_i log w_i
    s.t.      Σ w_i = 1,  Σ w_i μ_i ≥ μ_target,  w_i ≥ 0

The Lagrangian closed form is the Gibbs / softmax distribution:

    w_i = exp(β μ_i) / Σ_j exp(β μ_j)

β is the Lagrange multiplier dual to the return constraint:

  β → 0  : equal weight (maximum entropy)
  β → ∞  : winner-takes-all

We pick β NOT by val/test fitting but by a *structural* identity tied
to the portfolio config: choose β so the effective number of positions

    N_eff = 1 / Σ w_i²

equals `target_positions`. This is parameter-free in the sense that
`target_positions` is a portfolio-design choice that already exists in
`PortfolioConfig` (default 50), set long before any prediction is made.

The result is a long-only, fully-invested allocation whose
concentration is exactly what the operator declared up-front.
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


def _softmax_weights(signal: np.ndarray, beta: float) -> np.ndarray:
    """Numerically stable softmax of beta*signal."""
    z = beta * signal
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _effective_n(w: np.ndarray) -> float:
    s = float((w ** 2).sum())
    return 1.0 / s if s > 0 else float(len(w))


def _solve_beta(signal: np.ndarray, target_n_eff: float,
                lo: float = 0.0, hi: float = 200.0,
                tol: float = 0.5, max_iter: int = 40) -> float:
    """Binary search for β such that N_eff(softmax(β·signal)) ≈ target_n_eff."""
    n = len(signal)
    target_n_eff = float(max(2.0, min(target_n_eff, n)))

    # β=0 → uniform → N_eff = n
    # β→∞ → winner-takes-all → N_eff → 1
    # Effective N is monotonically decreasing in β.
    n_lo = _effective_n(_softmax_weights(signal, lo))
    n_hi = _effective_n(_softmax_weights(signal, hi))
    if n_hi >= target_n_eff:
        return hi
    if n_lo <= target_n_eff:
        return lo
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        n_mid = _effective_n(_softmax_weights(signal, mid))
        if abs(n_mid - target_n_eff) < tol:
            return mid
        if n_mid > target_n_eff:
            lo = mid  # need stronger concentration
        else:
            hi = mid
    return (lo + hi) / 2.0


def max_entropy_allocation(candidates: pd.DataFrame,
                           cfg: PortfolioConfig,
                           vol_estimator: VolEstimator | None = None,
                           as_of: date | None = None,
                           inverse_vol_blend: float = 0.5) -> AllocationResult:
    """Boltzmann-weighted long-only allocation.

    `inverse_vol_blend` interpolates between pure max-entropy on signals
    (blend=0.0) and a vol-adjusted variant where the effective signal is
    `signal − blend · log(σ_i / med(σ))`. The latter shrinks weights of
    high-vol names while preserving the entropy structure. blend=0.5 is
    a theoretical mid-point and is not searched over val/test data.
    """
    if candidates.empty:
        return AllocationResult(weights=pd.DataFrame(columns=[
            "internal_company_id", "weight", "predicted_signal", "vol"]),
            cash_weight=1.0 if cfg.allow_cash else 0.0,
            n_candidates=0)

    signal = candidates["positive_signal"].to_numpy().astype(float)
    vol = candidates["vol_estimate"].to_numpy().astype(float)

    if inverse_vol_blend > 0 and len(vol) > 0 and (vol > 0).any():
        med_vol = float(np.median(vol[vol > 0])) if (vol > 0).any() else 1.0
        med_vol = max(med_vol, 1e-9)
        log_vol_adj = np.log(np.maximum(vol, 1e-9) / med_vol)
        eff_signal = signal - inverse_vol_blend * log_vol_adj
    else:
        eff_signal = signal

    beta = _solve_beta(eff_signal, target_n_eff=cfg.target_positions)
    weights = _softmax_weights(eff_signal, beta)

    # Apply caps + min-weight filter, like the other allocators.
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


def build_max_entropy_target_portfolio(
    as_of: date,
    predictions_with_context: pd.DataFrame,
    vol_estimator: VolEstimator,
    cfg: PortfolioConfig,
    k: float = 2.5,
    sector_map: dict[str, str] | None = None,
    inverse_vol_blend: float = 0.5,
) -> AllocationResult:
    from afp.portfolio.candidate_selection import active_predictions
    from afp.portfolio.signal_restore import add_restored_returns

    active = active_predictions(predictions_with_context, as_of)
    if active.empty:
        return max_entropy_allocation(pd.DataFrame(), cfg)

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
    return max_entropy_allocation(candidates, cfg,
                                  vol_estimator=vol_estimator, as_of=as_of,
                                  inverse_vol_blend=inverse_vol_blend)
