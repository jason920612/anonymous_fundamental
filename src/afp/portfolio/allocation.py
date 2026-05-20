"""Signal-weighted inverse-vol allocation + caps + cash rule (RFC-06 §7-§12)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioConfig:
    min_positive_signal: float = 0.05
    signal_power: float = 1.0
    min_positions: int = 30
    target_positions: int = 50
    max_positions: int = 100
    max_single_stock_weight: float = 0.05
    min_position_weight: float = 0.0025
    allow_cash: bool = True
    leverage: float = 1.0
    max_sector_weight: float | None = None    # Phase 16: set to e.g. 0.30 to enable


@dataclass
class AllocationResult:
    weights: pd.DataFrame   # columns: internal_company_id, weight, predicted_signal, vol
    cash_weight: float
    n_candidates: int


def select_candidates(prediction_rows: pd.DataFrame, cfg: PortfolioConfig) -> pd.DataFrame:
    """Filter to long-only positive candidates with valid vol estimate."""
    df = prediction_rows.dropna(subset=["predicted_signal", "vol_estimate"]).copy()
    df["positive_signal"] = df["predicted_signal"].clip(lower=0.0)
    df = df[df["positive_signal"] >= cfg.min_positive_signal]
    if df.empty:
        return df
    sort_key = "positive_signal"
    df = df.sort_values(sort_key, ascending=False).head(cfg.max_positions)
    return df.reset_index(drop=True)


def _cap_weights(weights: np.ndarray, cap: float) -> np.ndarray:
    """Iteratively cap weights at `cap` and redistribute excess to non-capped names."""
    w = weights.copy()
    n = len(w)
    if n == 0:
        return w
    for _ in range(20):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = w[over].sum() - cap * over.sum()
        w[over] = cap
        under_mask = ~over
        if under_mask.sum() == 0 or w[under_mask].sum() == 0:
            break
        w[under_mask] += excess * w[under_mask] / w[under_mask].sum()
    return w


def _apply_sector_cap(weights: np.ndarray, sectors: np.ndarray, cap: float) -> np.ndarray:
    """Iteratively cap per-sector weight share. Excess flows to non-capped sectors."""
    w = weights.copy().astype(float)
    if len(w) == 0 or cap is None or cap >= 1.0:
        return w
    for _ in range(20):
        total = w.sum()
        if total <= 0:
            return w
        # sector-share matrix
        unique_sectors = pd.unique(sectors)
        excess = 0.0
        capped_sectors = []
        for s in unique_sectors:
            mask = sectors == s
            share = w[mask].sum() / total
            if share > cap + 1e-9:
                excess += (share - cap) * total
                scale = cap * total / w[mask].sum()
                w[mask] *= scale
                capped_sectors.append(s)
        if excess <= 1e-12:
            break
        free_mask = ~np.isin(sectors, capped_sectors) if capped_sectors else np.ones(len(w), dtype=bool)
        if free_mask.sum() == 0 or w[free_mask].sum() == 0:
            break
        w[free_mask] += excess * (w[free_mask] / w[free_mask].sum())
    return w


def inverse_vol_allocation(candidates: pd.DataFrame, cfg: PortfolioConfig) -> AllocationResult:
    if candidates.empty:
        return AllocationResult(weights=pd.DataFrame(columns=["internal_company_id", "weight",
                                                              "predicted_signal", "vol"]),
                                cash_weight=1.0 if cfg.allow_cash else 0.0,
                                n_candidates=0)

    rb_raw = candidates["positive_signal"].to_numpy() ** cfg.signal_power
    rb = rb_raw / rb_raw.sum() if rb_raw.sum() > 0 else rb_raw
    raw_w = rb / candidates["vol_estimate"].to_numpy()
    if raw_w.sum() <= 0:
        weights = np.full(len(candidates), 1.0 / len(candidates))
    else:
        weights = raw_w / raw_w.sum()

    # Drop tiny raw weights before capping so cap redistribution does not inflate them.
    weights = np.where(weights < cfg.min_position_weight, 0.0, weights)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    weights = _cap_weights(weights, cfg.max_single_stock_weight)
    # Sector cap (Phase 16) — optional, applied only if the candidate frame has a `sector` column
    if cfg.max_sector_weight is not None and "sector" in candidates.columns:
        sectors = candidates["sector"].fillna("UNK").to_numpy()
        weights = _apply_sector_cap(weights, sectors, cfg.max_sector_weight)
        weights = _cap_weights(weights, cfg.max_single_stock_weight)
    weight_sum = weights.sum()
    if weight_sum > cfg.leverage:
        weights = weights / weight_sum * cfg.leverage

    cash_weight = max(0.0, cfg.leverage - weights.sum())
    if not cfg.allow_cash and cash_weight > 1e-9 and weights.sum() > 0:
        # Force fully-invested by scaling up — clamp again to honor single-stock cap.
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
