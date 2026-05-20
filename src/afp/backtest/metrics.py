"""Performance metrics (RFC-07 §8)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(daily_returns: pd.Series, trading_days_per_year: int = 252) -> float:
    if daily_returns.empty:
        return 0.0
    final = float((1 + daily_returns).prod())
    n = len(daily_returns)
    return final ** (trading_days_per_year / n) - 1.0


def annualized_volatility(daily_returns: pd.Series, trading_days_per_year: int = 252) -> float:
    if len(daily_returns) < 2:
        return 0.0
    return float(daily_returns.std(ddof=1) * np.sqrt(trading_days_per_year))


def sharpe_ratio(daily_returns: pd.Series, risk_free_daily: float = 0.0,
                 trading_days_per_year: int = 252) -> float:
    vol = annualized_volatility(daily_returns, trading_days_per_year)
    if vol == 0:
        return 0.0
    return annualized_return(daily_returns - risk_free_daily, trading_days_per_year) / vol


def sortino_ratio(daily_returns: pd.Series, trading_days_per_year: int = 252) -> float:
    downside = daily_returns.clip(upper=0.0)
    dd = downside.std(ddof=1) * np.sqrt(trading_days_per_year)
    if dd == 0:
        return 0.0
    return annualized_return(daily_returns, trading_days_per_year) / dd


def max_drawdown(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return 0.0
    cum = (1 + daily_returns).cumprod()
    peak = cum.cummax()
    return float((cum / peak - 1.0).min())


def calmar_ratio(daily_returns: pd.Series, trading_days_per_year: int = 252) -> float:
    mdd = max_drawdown(daily_returns)
    if mdd == 0:
        return 0.0
    return annualized_return(daily_returns, trading_days_per_year) / abs(mdd)


def turnover(daily_turnover: pd.Series, trading_days_per_year: int = 252) -> float:
    if daily_turnover.empty:
        return 0.0
    return float(daily_turnover.mean()) * trading_days_per_year


def summary(daily_returns: pd.Series, daily_turnover: pd.Series | None = None,
            trading_days_per_year: int = 252) -> dict[str, float]:
    out = {
        "annualized_return": annualized_return(daily_returns, trading_days_per_year),
        "annualized_volatility": annualized_volatility(daily_returns, trading_days_per_year),
        "sharpe": sharpe_ratio(daily_returns, trading_days_per_year=trading_days_per_year),
        "sortino": sortino_ratio(daily_returns, trading_days_per_year),
        "max_drawdown": max_drawdown(daily_returns),
        "calmar": calmar_ratio(daily_returns, trading_days_per_year),
        "n_days": int(len(daily_returns)),
    }
    if daily_turnover is not None:
        out["annualized_turnover"] = turnover(daily_turnover, trading_days_per_year)
    return out
