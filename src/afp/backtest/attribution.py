"""Phase 11: portfolio attribution + concentration + cost-drag analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def top_contributors(holdings: pd.DataFrame, prices: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Sum of `weight_{d-1} * stock_return_d` per internal_company_id.

    `holdings` columns: date, internal_company_id, weight
    `prices` columns:   date, internal_company_id, adjusted_close
    """
    h = holdings.copy()
    h["date"] = pd.to_datetime(h["date"])
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])

    pivot = p.pivot_table(index="date", columns="internal_company_id",
                          values="adjusted_close", aggfunc="last").sort_index()
    daily_ret = pivot.pct_change().fillna(0.0)

    w_pivot = h.pivot_table(index="date", columns="internal_company_id",
                            values="weight", aggfunc="last").sort_index()
    w_pivot = w_pivot.reindex(daily_ret.index).ffill().fillna(0.0)
    w_yday = w_pivot.shift(1).fillna(0.0)

    common = w_yday.columns.intersection(daily_ret.columns)
    contribution = w_yday[common] * daily_ret[common]
    by_name = contribution.sum(axis=0).sort_values(ascending=False)
    return pd.DataFrame({"internal_company_id": by_name.index,
                         "total_contribution": by_name.values}).head(n)


def concentration(holdings: pd.DataFrame) -> pd.DataFrame:
    """Per-day Herfindahl index + top-5/top-10 weight share."""
    h = holdings.copy()
    h["date"] = pd.to_datetime(h["date"])
    out = []
    for d, grp in h.groupby("date"):
        w = grp["weight"].astype(float).sort_values(ascending=False)
        hhi = float((w ** 2).sum())
        out.append({
            "date": d.date(),
            "herfindahl": hhi,
            "top5_weight": float(w.head(5).sum()),
            "top10_weight": float(w.head(10).sum()),
            "n_positions": int(len(w)),
        })
    return pd.DataFrame(out)


def dollar_volume_exposure(holdings: pd.DataFrame, prices: pd.DataFrame,
                           lookback_days: int = 60) -> pd.DataFrame:
    """Per-day weighted average trailing dollar volume of held positions."""
    h = holdings.copy()
    h["date"] = pd.to_datetime(h["date"])
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["dollar_volume"] = p["adjusted_close"].astype(float) * p["volume"].astype(float)
    dv_pivot = p.pivot_table(index="date", columns="internal_company_id",
                             values="dollar_volume", aggfunc="last").sort_index()
    trailing = dv_pivot.rolling(lookback_days, min_periods=lookback_days // 4).mean()

    out = []
    for d, grp in h.groupby("date"):
        d_ts = pd.Timestamp(d)
        if d_ts not in trailing.index:
            continue
        row = trailing.loc[d_ts]
        weights = grp.set_index("internal_company_id")["weight"]
        common = weights.index.intersection(row.dropna().index)
        if len(common) == 0:
            out.append({"date": d.date(), "weighted_avg_dollar_volume": np.nan})
            continue
        w = weights.loc[common]
        v = row.loc[common]
        wavg = float((w * v).sum() / w.sum()) if w.sum() > 0 else np.nan
        out.append({"date": d.date(), "weighted_avg_dollar_volume": wavg})
    return pd.DataFrame(out)


def cash_and_turnover_summary(daily: pd.DataFrame, tcost_bps: float) -> dict:
    if daily.empty:
        return {}
    annual_turnover = float(daily["turnover"].mean()) * 252.0
    cost_drag = annual_turnover * 2 * (tcost_bps / 10000.0)
    return {
        "avg_cash_weight": float(daily["cash_weight"].mean()),
        "max_cash_weight": float(daily["cash_weight"].max()),
        "p95_cash_weight": float(daily["cash_weight"].quantile(0.95)),
        "annualized_turnover": annual_turnover,
        "cost_drag_per_year_pct": cost_drag * 100.0,
    }


def build_attribution_report(holdings: pd.DataFrame,
                             daily: pd.DataFrame,
                             prices: pd.DataFrame,
                             tcost_bps: float = 10.0) -> dict:
    return {
        "top_contributors": top_contributors(holdings, prices, n=20).to_dict(orient="records"),
        "concentration": concentration(holdings).to_dict(orient="records"),
        "dollar_volume_exposure": dollar_volume_exposure(holdings, prices).to_dict(orient="records"),
        "cash_turnover_summary": cash_and_turnover_summary(daily, tcost_bps),
    }
