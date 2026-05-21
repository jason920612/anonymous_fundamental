"""Cross-disciplinary market-state features (Phase 62).

Per-sample features computed at each `entry_date` reflecting the
market regime via cross-disciplinary observables:

- f_temp_z       — cross-sectional dispersion z-score (Phase 39, stat mech temperature)
- f_kuramoto_r   — Kuramoto coupled-oscillator order parameter (Phase 61)
- f_kuramoto_z   — rolling z-score of r
- f_gr_count_z   — Gutenberg-Richter foreshock-count z-score (Phase 60)
- f_market_mode  — λ_1 / trace of cleaned correlation matrix (RMT dominant eigenvalue)
- f_vov_z        — vol-of-vol z-score (universe-level, Phase 50)

These are MARKET-LEVEL features (same value for all samples on a given
entry_date). The LambdaRank model sees them as anonymous numeric
columns and can learn to condition its ranking on the market regime
without ever knowing the concept names — the anonymity invariant of
the project is preserved.

All features are computed CAUSALLY from data ≤ entry_date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CrossFeatureConfig:
    sigma_window: int = 60          # rolling vol window for per-stock σ
    short_vol_window: int = 20      # for vov inner
    history_window: int = 252       # rolling history for z-scores
    eigen_lookback: int = 252       # for market-mode eigenvalue


CROSS_FEATURE_IDS = [
    "f_market_temp_z",
    "f_kuramoto_r",
    "f_kuramoto_z",
    "f_gr_count_z",
    "f_market_mode_ratio",
    "f_vov_z",
]


def _cross_sectional_temperature(price_panel: pd.DataFrame) -> pd.Series:
    log_p = np.log(price_panel.astype(float).replace(0.0, np.nan))
    ret = log_p.diff()
    return ret.std(axis=1, ddof=1)


def _vov(price_panel: pd.DataFrame, sigma_window: int, vov_window: int) -> pd.Series:
    log_p = np.log(price_panel.astype(float).replace(0.0, np.nan))
    ret = log_p.diff()
    avg_abs = ret.abs().mean(axis=1)   # universe-mean absolute return per day
    sigma = avg_abs.rolling(sigma_window, min_periods=10).std(ddof=1)
    vov = sigma.rolling(vov_window, min_periods=10).std(ddof=1)
    return vov


def _kuramoto_r_panel(price_panel: pd.DataFrame, sigma_window: int) -> pd.Series:
    """Daily Kuramoto r over the universe — mirror of afp.portfolio.kuramoto."""
    log_p = np.log(price_panel.astype(float).replace(0.0, np.nan))
    ret = log_p.diff()
    sigma = ret.rolling(sigma_window, min_periods=20).std(ddof=1)
    r_values = []
    for idx in ret.index:
        row = ret.loc[idx].dropna()
        if len(row) < 10:
            r_values.append(np.nan)
            continue
        s_row = sigma.loc[idx].reindex(row.index)
        if s_row.isna().any():
            r_values.append(np.nan)
            continue
        s = np.where(s_row.to_numpy() > 1e-9, s_row.to_numpy(), 1.0)
        z = np.clip(row.to_numpy() / s, -1.0, 1.0)
        phi = np.pi * z
        r_values.append(float(np.abs(np.exp(1j * phi).mean())))
    return pd.Series(r_values, index=ret.index)


def _gr_count_z(daily_universe_ret: pd.Series, sigma_window: int, count_window: int,
                history_window: int) -> pd.Series:
    """Foreshock count z-score per day."""
    sigma = daily_universe_ret.rolling(sigma_window, min_periods=10).std(ddof=1)
    count_z = pd.Series(np.nan, index=daily_universe_ret.index)
    for i in range(history_window + count_window + sigma_window, len(daily_universe_ret) + 1):
        sub = daily_universe_ret.iloc[:i]
        s = sigma.iloc[i - 1]
        if not np.isfinite(s) or s <= 0:
            continue
        thresh = -s
        # rolling count over count_window
        counts = (sub < thresh).rolling(count_window, min_periods=10).sum()
        if counts.iloc[-1] is None or not np.isfinite(counts.iloc[-1]):
            continue
        history = counts.iloc[-(history_window + 1):-1].dropna()
        if len(history) < 30:
            continue
        mu = float(history.mean())
        sd = float(history.std(ddof=1))
        if sd < 1e-12:
            continue
        count_z.iloc[i - 1] = (counts.iloc[-1] - mu) / sd
    return count_z


def _market_mode_ratio(price_panel: pd.DataFrame, lookback: int,
                       stride: int = 21) -> pd.Series:
    """λ_1 / trace of correlation matrix on rolling window; subsampled
    every `stride` days then forward-filled. Eigendecomposition on a
    ~1000-asset matrix is expensive (~1-2s each), so we recompute
    only ~monthly and let the structural quantity drift in between.

    Uses scipy.sparse.linalg.eigsh to get only the top eigenvalue
    (fast — O(N²) instead of O(N³)).
    """
    log_p = np.log(price_panel.astype(float).replace(0.0, np.nan))
    ret = log_p.diff()
    out = pd.Series(np.nan, index=ret.index)
    if len(ret) < lookback:
        return out
    try:
        from scipy.sparse.linalg import eigsh
    except ImportError:
        eigsh = None
    for i in range(lookback, len(ret) + 1, stride):
        window = ret.iloc[i - lookback:i].dropna(axis=1, how="any")
        if window.shape[1] < 10:
            continue
        try:
            corr = window.corr().to_numpy()
            corr = np.nan_to_num(corr, nan=0.0)
            if eigsh is not None and corr.shape[0] > 50:
                lam_max = float(eigsh(corr, k=1, which="LA",
                                       return_eigenvectors=False)[0])
            else:
                lam_max = float(np.linalg.eigvalsh(corr)[-1])
            out.iloc[i - 1] = lam_max / max(float(np.trace(corr)), 1e-9)
        except Exception:
            continue
    return out.ffill()


def compute_cross_features(price_panel: pd.DataFrame,
                           cfg: CrossFeatureConfig | None = None) -> pd.DataFrame:
    """Build daily DataFrame of cross-disciplinary market features.

    Returns DataFrame indexed by date with one column per feature in
    `CROSS_FEATURE_IDS`. NaN before sufficient history accumulates.
    """
    cfg = cfg or CrossFeatureConfig()
    temp = _cross_sectional_temperature(price_panel)
    temp_mu = temp.rolling(cfg.history_window, min_periods=20).mean()
    temp_sd = temp.rolling(cfg.history_window, min_periods=20).std(ddof=1)
    temp_z = (temp - temp_mu) / temp_sd.replace(0.0, np.nan)

    r_series = _kuramoto_r_panel(price_panel, cfg.sigma_window)
    r_mu = r_series.rolling(cfg.history_window, min_periods=30).mean()
    r_sd = r_series.rolling(cfg.history_window, min_periods=30).std(ddof=1)
    r_z = (r_series - r_mu) / r_sd.replace(0.0, np.nan)

    daily_universe_ret = np.log(price_panel.astype(float).replace(0.0, np.nan)).diff().mean(axis=1)
    gr_z = _gr_count_z(daily_universe_ret, cfg.sigma_window, cfg.sigma_window,
                       cfg.history_window)

    mode = _market_mode_ratio(price_panel, cfg.eigen_lookback)

    vov = _vov(price_panel, cfg.short_vol_window, cfg.sigma_window)
    vov_mu = vov.rolling(cfg.history_window, min_periods=30).mean()
    vov_sd = vov.rolling(cfg.history_window, min_periods=30).std(ddof=1)
    vov_z = (vov - vov_mu) / vov_sd.replace(0.0, np.nan)

    out = pd.DataFrame(index=price_panel.index)
    out["f_market_temp_z"] = temp_z
    out["f_kuramoto_r"] = r_series
    out["f_kuramoto_z"] = r_z
    out["f_gr_count_z"] = gr_z
    out["f_market_mode_ratio"] = mode
    out["f_vov_z"] = vov_z
    return out


def attach_cross_features_to_samples(samples: pd.DataFrame,
                                     feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Join market-state features to samples by `entry_date`."""
    samples = samples.copy()
    samples["entry_date"] = pd.to_datetime(samples["entry_date"])
    # Use the last AVAILABLE feature value AT OR BEFORE entry_date (causal)
    feat = feature_panel.copy()
    feat.index = pd.to_datetime(feat.index)
    feat = feat.sort_index()
    # reindex to sample dates by forward-filling from past
    samples_sorted = samples.sort_values("entry_date").reset_index(drop=True)
    feat_at_samples = feat.reindex(samples_sorted["entry_date"], method="ffill")
    for col in CROSS_FEATURE_IDS:
        samples_sorted[col] = feat_at_samples[col].to_numpy()
        samples_sorted[f"{col}_missing"] = samples_sorted[col].isna().astype(float)
        samples_sorted[col] = samples_sorted[col].fillna(0.0)
    return samples_sorted
