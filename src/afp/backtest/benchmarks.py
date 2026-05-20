"""Benchmark daily-return series (RFC-07 §7).

Includes the four canonical benchmarks (SPY, QQQ, equal-weight universe,
equal-risk universe) plus Phase 11 additions: dollar-volume-weighted universe,
random-positive-signal, shuffled-signal, and "active model universe" variants
that restrict to names with at least one valid prediction in the test window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def benchmark_daily_returns(benchmarks: pd.DataFrame, symbol: str) -> pd.Series:
    """Simple daily returns (pct change) for an adjusted-close benchmark."""
    df = benchmarks[benchmarks["ticker_at_date"] == symbol].sort_values("date")
    px = df["adjusted_close"].astype(float)
    rets = px.pct_change().fillna(0.0)
    rets.index = pd.to_datetime(df["date"].to_numpy())
    return rets


def equal_weight_universe_returns(prices: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Daily equal-weighted return across all companies active on each date."""
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="internal_company_id",
                           values="adjusted_close", aggfunc="last")
    pivot = pivot.reindex(dates).ffill()
    rets = pivot.pct_change().fillna(0.0)
    return rets.mean(axis=1)


def equal_risk_universe_returns(prices: pd.DataFrame, dates: pd.DatetimeIndex,
                                lookback_days: int = 252) -> pd.Series:
    """Daily inverse-vol weighted return across all companies (no model signal)."""
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="internal_company_id",
                           values="adjusted_close", aggfunc="last")
    pivot = pivot.reindex(dates).ffill()
    daily_rets = pivot.pct_change().fillna(0.0)
    rolling_vol = daily_rets.rolling(lookback_days, min_periods=lookback_days // 4).std().shift(1)
    inv_vol = (1.0 / rolling_vol).replace([np.inf, -np.inf], np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0.0)
    return (weights * daily_rets).sum(axis=1)


# ---------------------------------------------------------------------------
# Phase 11 additions
# ---------------------------------------------------------------------------

def dollar_volume_weighted_universe_returns(prices: pd.DataFrame, dates: pd.DatetimeIndex,
                                            lookback_days: int = 252) -> pd.Series:
    """Cap-proxy: each name weighted by its trailing avg dollar volume."""
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dollar_volume"] = df["adjusted_close"].astype(float) * df["volume"].astype(float)
    px = df.pivot_table(index="date", columns="internal_company_id",
                        values="adjusted_close", aggfunc="last").reindex(dates).ffill()
    dv = df.pivot_table(index="date", columns="internal_company_id",
                        values="dollar_volume", aggfunc="last").reindex(dates).ffill()
    daily_rets = px.pct_change().fillna(0.0)
    trailing_dv = dv.rolling(lookback_days, min_periods=lookback_days // 4).mean().shift(1)
    weights = trailing_dv.div(trailing_dv.sum(axis=1), axis=0).fillna(0.0)
    return (weights * daily_rets).sum(axis=1)


def active_universe_mask(predictions: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean matrix `[dates, internal_company_id]` of names whose signal is active on each date."""
    preds = predictions.dropna(subset=["entry_date", "exit_date"]).copy()
    preds["entry_date"] = pd.to_datetime(preds["entry_date"])
    preds["exit_date"] = pd.to_datetime(preds["exit_date"])
    companies = preds["internal_company_id"].unique()
    out = pd.DataFrame(False, index=dates, columns=companies)
    for icid, grp in preds.groupby("internal_company_id"):
        for row in grp.itertuples():
            mask = (dates >= row.entry_date) & (dates <= row.exit_date)
            out.loc[mask, icid] = True
    return out


def equal_weight_active_universe_returns(prices: pd.DataFrame, predictions: pd.DataFrame,
                                         dates: pd.DatetimeIndex) -> pd.Series:
    """Each day, equal weight across names with an active prediction."""
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="internal_company_id",
                           values="adjusted_close", aggfunc="last").reindex(dates).ffill()
    daily_rets = pivot.pct_change().fillna(0.0)
    mask = active_universe_mask(predictions, dates)
    mask = mask.reindex(columns=pivot.columns, fill_value=False)
    n_active = mask.sum(axis=1).replace(0, np.nan)
    weights = mask.astype(float).div(n_active, axis=0).fillna(0.0)
    return (weights * daily_rets).sum(axis=1)


def equal_risk_active_universe_returns(prices: pd.DataFrame, predictions: pd.DataFrame,
                                       dates: pd.DatetimeIndex,
                                       lookback_days: int = 252,
                                       transaction_cost_bps: float = 0.0) -> pd.Series:
    """Inverse-vol weighted across names with an active prediction.

    If `transaction_cost_bps > 0`, the daily turnover is deducted from the
    return (single-sided turnover * bps / 10000). The cost-free version (=0)
    is the unrealistic baseline most academic papers quote; the cost-aware
    version (~10 bps) is what a real fund would actually realize.
    """
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="internal_company_id",
                           values="adjusted_close", aggfunc="last").reindex(dates).ffill()
    daily_rets = pivot.pct_change().fillna(0.0)
    rolling_vol = daily_rets.rolling(lookback_days, min_periods=lookback_days // 4).std().shift(1)
    inv_vol = (1.0 / rolling_vol).replace([np.inf, -np.inf], np.nan)

    mask = active_universe_mask(predictions, dates)
    mask = mask.reindex(columns=pivot.columns, fill_value=False)
    masked = inv_vol.where(mask, other=np.nan)
    weights = masked.div(masked.sum(axis=1), axis=0).fillna(0.0)
    gross = (weights * daily_rets).sum(axis=1)
    if transaction_cost_bps > 0:
        turnover = (weights.diff().abs().sum(axis=1) * 0.5).fillna(0.0)
        gross = gross - turnover * (transaction_cost_bps / 10000.0)
    return gross


def random_positive_signal_returns(prices: pd.DataFrame, predictions: pd.DataFrame,
                                   dates: pd.DatetimeIndex,
                                   n_holdings: int = 50,
                                   seed: int = 42,
                                   lookback_days: int = 252) -> pd.Series:
    """Each day, pick `n_holdings` random active names; weight inverse-vol.

    This is the apples-to-apples random baseline for the long-only strategy:
    same N positions, same inverse-vol allocation, but signals are uniform over
    the active set instead of model-driven.
    """
    rng = np.random.default_rng(seed)
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="internal_company_id",
                           values="adjusted_close", aggfunc="last").reindex(dates).ffill()
    daily_rets = pivot.pct_change().fillna(0.0)
    rolling_vol = daily_rets.rolling(lookback_days, min_periods=lookback_days // 4).std().shift(1)
    inv_vol = (1.0 / rolling_vol).replace([np.inf, -np.inf], np.nan)

    mask = active_universe_mask(predictions, dates).reindex(columns=pivot.columns, fill_value=False)
    out = pd.Series(0.0, index=dates)
    for d in dates:
        active = mask.loc[d]
        candidates = active[active].index.to_numpy()
        if len(candidates) == 0:
            continue
        picks = rng.choice(candidates, size=min(n_holdings, len(candidates)), replace=False)
        iv = inv_vol.loc[d, picks].dropna()
        if iv.empty:
            continue
        w = iv / iv.sum()
        out.loc[d] = float((w * daily_rets.loc[d, w.index]).sum())
    return out


def shuffled_signal_returns(prices: pd.DataFrame, predictions: pd.DataFrame,
                            dates: pd.DatetimeIndex,
                            n_holdings: int = 50,
                            seed: int = 42,
                            lookback_days: int = 252) -> pd.Series:
    """Use the model's predicted signals but shuffle them across companies on each day.

    Tests whether the *magnitude distribution* of signals or the *attribution to
    specific companies* is what drives PnL. If shuffled performs nearly as well,
    the model is not adding company-specific information.
    """
    rng = np.random.default_rng(seed)
    preds = predictions.dropna(subset=["entry_date", "exit_date", "predicted_signal"]).copy()
    preds["entry_date"] = pd.to_datetime(preds["entry_date"])
    preds["exit_date"] = pd.to_datetime(preds["exit_date"])

    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="internal_company_id",
                           values="adjusted_close", aggfunc="last").reindex(dates).ffill()
    daily_rets = pivot.pct_change().fillna(0.0)
    rolling_vol = daily_rets.rolling(lookback_days, min_periods=lookback_days // 4).std().shift(1)
    inv_vol = (1.0 / rolling_vol).replace([np.inf, -np.inf], np.nan)

    out = pd.Series(0.0, index=dates)
    for d in dates:
        active = preds[(preds["entry_date"] <= d) & (preds["exit_date"] >= d)]
        if active.empty:
            continue
        shuffled = active.copy()
        signals = active["predicted_signal"].to_numpy()
        rng.shuffle(signals)
        shuffled["predicted_signal"] = signals
        shuffled = shuffled[shuffled["predicted_signal"] > 0]
        if shuffled.empty:
            continue
        shuffled = shuffled.sort_values("predicted_signal", ascending=False).head(n_holdings)
        iv = inv_vol.loc[d].reindex(shuffled["internal_company_id"]).dropna()
        if iv.empty:
            continue
        risk_budget = shuffled.set_index("internal_company_id").loc[iv.index, "predicted_signal"]
        raw_w = risk_budget / iv
        if raw_w.sum() <= 0:
            continue
        w = raw_w / raw_w.sum()
        out.loc[d] = float((w * daily_rets.loc[d, w.index]).sum())
    return out

