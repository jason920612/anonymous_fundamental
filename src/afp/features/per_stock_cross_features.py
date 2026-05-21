"""Per-stock cross-disciplinary features (Phase 64).

Builds on Phase 62 (market-state features) by adding PER-STOCK
features computed from each stock's own recent return series. These
give the model new dimensions to discriminate between candidates
beyond the existing anonymous fundamentals + price features.

Features per (sample_id, entry_date, internal_company_id):

- `f_stock_hurst`     — Hurst exponent (R/S analysis, Mandelbrot)
                        H > 0.5 = trending, H = 0.5 = random walk,
                        H < 0.5 = mean-reverting
- `f_stock_skew`      — 252d rolling return skew (econophysics)
- `f_stock_kurtosis`  — 252d rolling return kurtosis (excess)
- `f_stock_levy_z`    — frequency of >3σ events vs 0.27% expected
                        (Lévy / heavy-tail diagnostic)
- `f_stock_lyapunov`  — finite-time max-Lyapunov-like exponent
                        (chaos theory; positive = sensitive)
- `f_stock_dfa_alpha` — detrended fluctuation analysis exponent
                        (long-range correlation)

All anonymous; all causally computed from prices ≤ entry_date.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


PER_STOCK_FEATURE_IDS = [
    "f_stock_hurst",
    "f_stock_skew",
    "f_stock_kurtosis",
    "f_stock_levy_z",
    "f_stock_lyapunov",
    "f_stock_dfa_alpha",
]


@dataclass
class PerStockFeatureConfig:
    lookback_days: int = 252
    levy_sigma_threshold: float = 3.0
    expected_levy_freq: float = 0.0027   # P(|Z|>3) for Gaussian


def _hurst(series: np.ndarray) -> float:
    """R/S analysis estimator of the Hurst exponent."""
    if len(series) < 50:
        return 0.5
    series = series[~np.isnan(series)]
    n = len(series)
    if n < 50:
        return 0.5
    lags = [10, 20, 50]
    if n >= 200:
        lags.extend([100, 200])
    rs_values = []
    for L in lags:
        if L >= n:
            continue
        rs_per_block = []
        for start in range(0, n - L, L):
            chunk = series[start:start + L]
            mean = chunk.mean()
            dev = chunk - mean
            cum = np.cumsum(dev)
            R = cum.max() - cum.min()
            S = chunk.std(ddof=1)
            if S > 1e-12 and R > 0:
                rs_per_block.append(R / S)
        if rs_per_block:
            rs_values.append((L, np.mean(rs_per_block)))
    if len(rs_values) < 2:
        return 0.5
    Ls = np.log([x[0] for x in rs_values])
    RSs = np.log([x[1] for x in rs_values])
    slope, _ = np.polyfit(Ls, RSs, 1)
    return float(np.clip(slope, 0.0, 1.0))


def _levy_z(series: np.ndarray, sigma_threshold: float,
            expected_freq: float) -> float:
    """Z-score of observed frequency of |return| > k·σ vs Gaussian expectation."""
    if len(series) < 50:
        return 0.0
    sigma = float(np.std(series, ddof=1))
    if sigma <= 0:
        return 0.0
    threshold = sigma_threshold * sigma
    observed = float(np.mean(np.abs(series) > threshold))
    n = len(series)
    # Standard error of a Bernoulli proportion
    se = max(np.sqrt(expected_freq * (1 - expected_freq) / n), 1e-9)
    return (observed - expected_freq) / se


def _max_lyapunov(series: np.ndarray) -> float:
    """Finite-time max-Lyapunov-like exponent: mean log absolute return rate."""
    if len(series) < 50:
        return 0.0
    abs_ret = np.abs(series)
    abs_ret = abs_ret[abs_ret > 1e-9]
    if len(abs_ret) < 10:
        return 0.0
    return float(np.mean(np.log(abs_ret)))


def _dfa_alpha(series: np.ndarray) -> float:
    """Detrended fluctuation analysis (DFA) exponent.

    α = 0.5 random walk; α > 0.5 long-range persistence; α < 0.5
    anti-persistence.
    """
    n = len(series)
    if n < 64:
        return 0.5
    y = np.cumsum(series - series.mean())
    box_sizes = [8, 16, 32, 64]
    if n >= 128:
        box_sizes.append(128)
    F_n = []
    for s in box_sizes:
        if s >= n:
            continue
        nb = n // s
        if nb < 2:
            continue
        rms_blocks = []
        for i in range(nb):
            seg = y[i * s:(i + 1) * s]
            xs = np.arange(s)
            coef = np.polyfit(xs, seg, 1)
            trend = np.polyval(coef, xs)
            rms_blocks.append(np.sqrt(np.mean((seg - trend) ** 2)))
        if rms_blocks:
            F_n.append((s, np.mean(rms_blocks)))
    if len(F_n) < 2:
        return 0.5
    xs = np.log([x[0] for x in F_n])
    ys = np.log([max(x[1], 1e-12) for x in F_n])
    slope, _ = np.polyfit(xs, ys, 1)
    return float(np.clip(slope, 0.0, 1.5))


def compute_per_stock_features(samples: pd.DataFrame, prices: pd.DataFrame,
                                cfg: PerStockFeatureConfig | None = None) -> pd.DataFrame:
    """For each sample, compute its 6 per-stock cross-disciplinary features
    from the stock's recent (≤entry_date) daily returns.

    Returns DataFrame with sample_id + the 6 feature columns + their missing flags.
    """
    cfg = cfg or PerStockFeatureConfig()
    samples = samples.copy()
    samples["entry_date"] = pd.to_datetime(samples["entry_date"])
    prices_sorted = prices.copy()
    prices_sorted["date"] = pd.to_datetime(prices_sorted["date"])
    prices_sorted = prices_sorted.sort_values(["internal_company_id", "date"])

    by_company: dict[str, pd.DataFrame] = {
        cid: grp[["date", "adjusted_close"]]
        for cid, grp in prices_sorted.groupby("internal_company_id")
    }

    out_rows = []
    for _, row in samples.iterrows():
        icid = row["internal_company_id"]
        as_of = row["entry_date"]
        cf = by_company.get(icid)
        feats = {fid: 0.0 for fid in PER_STOCK_FEATURE_IDS}
        miss = {f"{fid}_missing": 1.0 for fid in PER_STOCK_FEATURE_IDS}
        if cf is not None:
            past = cf[cf["date"] < as_of].tail(cfg.lookback_days + 1)
            if len(past) >= 50:
                p = past["adjusted_close"].astype(float).to_numpy()
                p = p[p > 0]
                if len(p) >= 2:
                    ret = np.diff(np.log(p))
                    feats["f_stock_hurst"] = _hurst(ret)
                    feats["f_stock_skew"] = float(skew(ret, nan_policy="omit"))
                    feats["f_stock_kurtosis"] = float(kurtosis(ret, nan_policy="omit",
                                                                 fisher=True))
                    feats["f_stock_levy_z"] = _levy_z(ret, cfg.levy_sigma_threshold,
                                                       cfg.expected_levy_freq)
                    feats["f_stock_lyapunov"] = _max_lyapunov(ret)
                    feats["f_stock_dfa_alpha"] = _dfa_alpha(ret)
                    for fid in PER_STOCK_FEATURE_IDS:
                        miss[f"{fid}_missing"] = 0.0
        rec = {"sample_id": row["sample_id"]}
        rec.update(feats)
        rec.update(miss)
        out_rows.append(rec)

    return pd.DataFrame(out_rows)
