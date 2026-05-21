"""Hierarchical Risk Parity allocator (Phase 42 candidate).

López de Prado 2016: instead of inverting a noisy covariance matrix
(which amplifies noise eigendirections), build a *hierarchical tree*
of assets via correlation-distance clustering, then apply risk-parity
recursively down the tree. The procedure has no matrix inversion at
all, so it is naturally robust to the noise that breaks Markowitz.

Physics interpretation: the correlation-distance
`d_ij = √(½(1 - ρ_ij))` is a *proper metric* on the assets, and
single-linkage clustering on this metric is equivalent to building
the minimum spanning tree (MST) of the correlation graph. The MST is
the skeleton of pairwise dependencies — discarding all redundant
edges. This is the same logic by which percolation theory extracts
the critical backbone of a network.

The recursive bisection then allocates risk-parity weights down each
branch of the tree, naturally clustering correlated assets together
and avoiding the "lump risk on one factor" failure mode of naive
inverse-vol.

No hyperparameters that need val/test tuning.
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
from afp.portfolio.rmt_covariance import clean_covariance_rmt


def _corr_distance(corr: np.ndarray) -> np.ndarray:
    """López de Prado's correlation-distance metric: d = √(½(1 - ρ))."""
    d = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    np.fill_diagonal(d, 0.0)
    return d


def _quasi_diagonal_order(linkage: np.ndarray, n: int) -> list[int]:
    """Recursively recover the leaf order from a SciPy-style linkage matrix."""
    # linkage rows: (left, right, dist, count) ; new node id = n + i
    order = [int(linkage[-1, 0]), int(linkage[-1, 1])]
    while max(order) >= n:
        new_order = []
        for x in order:
            if x < n:
                new_order.append(x)
            else:
                row = linkage[int(x) - n]
                new_order.extend([int(row[0]), int(row[1])])
        order = new_order
    return order


def _single_linkage(distance: np.ndarray) -> np.ndarray:
    """Pure-numpy single-linkage agglomerative clustering on a distance matrix.

    Returns a SciPy-compatible linkage matrix (n-1 rows).
    """
    n = distance.shape[0]
    # Active cluster set: ids 0..n-1 initially; new clusters get id n, n+1, ...
    active = list(range(n))
    sizes = {i: 1 for i in active}
    # Cluster id → flat list of original assets
    members = {i: [i] for i in active}

    # Pairwise distances between active clusters; start from the leaf distance matrix
    dmat = distance.copy()
    next_id = n
    linkage_rows = []
    while len(active) > 1:
        # Find the closest pair among active clusters.
        # Work in indices of `active`
        m = len(active)
        # Get a view of pairwise distances among active.
        idx = np.array(active)
        sub = dmat[np.ix_(idx, idx)]
        np.fill_diagonal(sub, np.inf)
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        if i > j:
            i, j = j, i
        a, b = active[i], active[j]
        d_ab = sub[i, j]
        new_size = sizes[a] + sizes[b]
        linkage_rows.append([a, b, d_ab, new_size])
        new_members = members[a] + members[b]
        members[next_id] = new_members
        sizes[next_id] = new_size
        # Single linkage: dist(new, k) = min(dist(a, k), dist(b, k))
        new_dists = np.minimum(dmat[a], dmat[b])
        # Grow dmat
        dmat = np.pad(dmat, ((0, 1), (0, 1)), mode="constant", constant_values=np.inf)
        for k in active:
            if k == a or k == b:
                continue
            dmat[next_id, k] = new_dists[k]
            dmat[k, next_id] = new_dists[k]
        dmat[next_id, next_id] = 0.0
        active = [c for c in active if c != a and c != b] + [next_id]
        next_id += 1
    return np.array(linkage_rows)


def _recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """Allocate weights via top-down recursive risk-parity bisection."""
    n = cov.shape[0]
    w = np.ones(n)
    clusters = [order]
    while clusters:
        new_clusters = []
        for cl in clusters:
            if len(cl) < 2:
                continue
            mid = len(cl) // 2
            left, right = cl[:mid], cl[mid:]
            # Compute inverse-variance "cluster variances"
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1 - var_left / (var_left + var_right + 1e-12)
            # Equivalently: alpha = var_right / (var_left + var_right)
            for i in left:
                w[i] *= alpha
            for j in right:
                w[j] *= (1 - alpha)
            new_clusters.append(left)
            new_clusters.append(right)
        clusters = new_clusters
    return w


def _cluster_variance(cov: np.ndarray, ids: list[int]) -> float:
    """Inverse-variance-weighted internal variance of a cluster."""
    sub_cov = cov[np.ix_(ids, ids)]
    inv_var = 1.0 / np.maximum(np.diag(sub_cov), 1e-12)
    w_ivp = inv_var / inv_var.sum()
    return float(w_ivp @ sub_cov @ w_ivp)


def hrp_weights(cov: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity weights for a (cleaned) covariance matrix."""
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    sigma = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(sigma, sigma)
    np.fill_diagonal(corr, 1.0)
    distance = _corr_distance(corr)
    linkage = _single_linkage(distance)
    order = _quasi_diagonal_order(linkage, n)
    # The order may miss some leaves in edge cases — backfill.
    seen = set(order)
    for i in range(n):
        if i not in seen:
            order.append(i)
    w = _recursive_bisection(cov, order)
    if w.sum() > 0:
        w = w / w.sum()
    return w


def hrp_allocation(candidates: pd.DataFrame, cfg: PortfolioConfig,
                   vol_estimator: VolEstimator | None = None,
                   as_of: date | None = None,
                   lookback_days: int = 252,
                   signal_tilt_beta: float = 5.0) -> AllocationResult:
    """HRP weights blended with a softmax signal tilt.

    Pure HRP ignores μ. To respect the long-only signal we element-wise
    multiply HRP weights by `exp(β·μ̂)` and renormalize. β=5 is the
    same theoretical choice as Phase 38.
    """
    if candidates.empty:
        return AllocationResult(weights=pd.DataFrame(columns=[
            "internal_company_id", "weight", "predicted_signal", "vol"]),
            cash_weight=1.0 if cfg.allow_cash else 0.0,
            n_candidates=0)
    ids = candidates["internal_company_id"].tolist()

    from afp.portfolio.rmt_allocator import _build_returns_panel
    returns_panel = None
    if vol_estimator is not None and as_of is not None:
        returns_panel = _build_returns_panel(ids, as_of, vol_estimator, lookback_days)
    if returns_panel is None or returns_panel.shape[1] < 2:
        from afp.portfolio.allocation import inverse_vol_allocation
        return inverse_vol_allocation(candidates, cfg)

    cov = clean_covariance_rmt(returns_panel)
    w_hrp = hrp_weights(cov)

    signal = candidates["positive_signal"].to_numpy().astype(float)
    if signal.std() > 1e-9 and signal_tilt_beta > 0:
        tilt = np.exp(signal_tilt_beta * (signal - signal.max()))
        w = w_hrp * tilt
    else:
        w = w_hrp
    if w.sum() > 0:
        w = w / w.sum()

    w = np.where(w < cfg.min_position_weight, 0.0, w)
    if w.sum() > 0:
        w = w / w.sum()
    w = _cap_weights(w, cfg.max_single_stock_weight)

    if cfg.max_sector_weight is not None and "sector" in candidates.columns:
        sectors = candidates["sector"].fillna("UNK").to_numpy()
        w = _apply_sector_cap(w, sectors, cfg.max_sector_weight)
        w = _cap_weights(w, cfg.max_single_stock_weight)

    if w.sum() > cfg.leverage:
        w = w / w.sum() * cfg.leverage
    cash_weight = max(0.0, cfg.leverage - w.sum())
    if not cfg.allow_cash and cash_weight > 1e-9 and w.sum() > 0:
        w = w / w.sum() * cfg.leverage
        w = _cap_weights(w, cfg.max_single_stock_weight)
        cash_weight = max(0.0, cfg.leverage - w.sum())

    out = pd.DataFrame({
        "internal_company_id": candidates["internal_company_id"].to_numpy(),
        "weight": w,
        "predicted_signal": candidates["predicted_signal"].to_numpy(),
        "vol": candidates["vol_estimate"].to_numpy(),
    })
    out = out[out["weight"] > 0].reset_index(drop=True)
    return AllocationResult(weights=out, cash_weight=float(cash_weight),
                            n_candidates=len(candidates))


def build_hrp_target_portfolio(
    as_of: date,
    predictions_with_context: pd.DataFrame,
    vol_estimator: VolEstimator,
    cfg: PortfolioConfig,
    k: float = 2.5,
    sector_map: dict[str, str] | None = None,
    lookback_days: int = 252,
    signal_tilt_beta: float = 5.0,
) -> AllocationResult:
    from afp.portfolio.candidate_selection import active_predictions
    from afp.portfolio.signal_restore import add_restored_returns

    active = active_predictions(predictions_with_context, as_of)
    if active.empty:
        return hrp_allocation(pd.DataFrame(), cfg)

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
    return hrp_allocation(candidates, cfg, vol_estimator=vol_estimator, as_of=as_of,
                          lookback_days=lookback_days,
                          signal_tilt_beta=signal_tilt_beta)
