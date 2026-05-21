"""Event-driven daily backtest engine (RFC-07 §3-§5)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from afp.data.trading_calendar import TradingCalendar
from afp.portfolio.allocation import PortfolioConfig
from afp.portfolio.candidate_selection import build_target_portfolio
from afp.portfolio.risk_models import RiskConfig, VolEstimator


@dataclass
class BacktestConfig:
    start_date: str
    end_date: str | None
    initial_value: float = 1.0
    transaction_cost_bps_per_trade: float = 10.0
    cash_return_daily: float = 0.0


@dataclass
class BacktestResult:
    daily: pd.DataFrame      # date, gross, cost, net, value, turnover, cash, n_positions
    holdings: pd.DataFrame   # date, internal_company_id, weight
    trades: pd.DataFrame     # date, internal_company_id, old_weight, new_weight, trade_weight, trade_cost, reason


# ---------------------------------------------------------------------------

def _price_pivot(prices: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="internal_company_id",
                           values="adjusted_close", aggfunc="last")
    return pivot.reindex(dates).ffill()


def _daily_simple_returns(price_pivot: pd.DataFrame) -> pd.DataFrame:
    return price_pivot.pct_change().fillna(0.0)


def run_backtest(
    predictions_with_context: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: TradingCalendar,
    backtest_cfg: BacktestConfig,
    portfolio_cfg: PortfolioConfig,
    risk_cfg: RiskConfig | None = None,
    k: float = 2.5,
    sector_map: dict[str, str] | None = None,
    distribution_cfg=None,
    vol_target_cfg=None,
    custom_allocator=None,
    regime_filter_cfg=None,
    regime_temperature_series=None,
    vol_estimator: VolEstimator | None = None,
    dd_target_cfg=None,
    beta_target_cfg=None,
    beta_market_returns: pd.Series | None = None,
    vov_target_cfg=None,
    gr_cfg=None,
    kuramoto_cfg=None,
    kuramoto_r_series=None,
) -> BacktestResult:
    risk_cfg = risk_cfg or RiskConfig()
    vol_est = vol_estimator if vol_estimator is not None else VolEstimator(prices, risk_cfg)
    vol_target_history: list[float] = []
    vol_target_prev = 1.0
    dd_target_prev = 1.0
    beta_target_prev = 1.0
    vov_target_prev = 1.0
    gr_prev = 1.0
    kuramoto_prev = 1.0

    start = pd.Timestamp(backtest_cfg.start_date)
    end = pd.Timestamp(backtest_cfg.end_date) if backtest_cfg.end_date else \
          pd.Timestamp(calendar.trading_dates[-1])
    dates = calendar.trading_dates[(calendar.trading_dates >= start) & (calendar.trading_dates <= end)]
    if len(dates) == 0:
        raise ValueError("no trading days in backtest window")

    price_pivot = _price_pivot(prices, dates)
    returns = _daily_simple_returns(price_pivot)

    tcost_rate = backtest_cfg.transaction_cost_bps_per_trade / 10000.0

    current_weights: dict[str, float] = {}
    cash_weight = 1.0
    portfolio_value = backtest_cfg.initial_value

    daily_rows = []
    holdings_rows = []
    trades_rows = []

    # Pre-compute set of dates that have at least one new signal so we know when to rebalance
    pred_e = pd.to_datetime(predictions_with_context["entry_date"])
    pred_x = pd.to_datetime(predictions_with_context["exit_date"])
    triggers = set(pd.to_datetime(pred_e).dt.normalize().tolist())
    triggers |= set(pd.to_datetime(pred_x).dt.normalize().tolist())

    prev_date = None
    for d in dates:
        d_norm = pd.Timestamp(d).normalize()
        # 1. Apply weight drift over the previous day's return
        if prev_date is not None and current_weights:
            r = returns.loc[d]
            new_weights: dict[str, float] = {}
            equity_factor = 0.0
            for icid, w in current_weights.items():
                stock_ret = float(r.get(icid, 0.0)) if icid in returns.columns else 0.0
                new_weights[icid] = w * (1.0 + stock_ret)
                equity_factor += new_weights[icid] - w
            cash_factor = cash_weight * backtest_cfg.cash_return_daily
            gross_return = equity_factor + cash_factor
            cash_weight = cash_weight * (1 + backtest_cfg.cash_return_daily)
            current_weights = new_weights
            total = sum(current_weights.values()) + cash_weight
            if total > 0:
                for icid in list(current_weights):
                    current_weights[icid] /= total
                cash_weight /= total
        else:
            gross_return = 0.0

        # 2. Trigger rebalance if today is an entry/exit date
        cost = 0.0
        turnover = 0.0
        if d_norm in triggers or not current_weights:
            if custom_allocator is not None:
                target = custom_allocator(
                    d.date(), predictions_with_context, vol_est,
                    portfolio_cfg, k, sector_map)
            elif distribution_cfg is not None and getattr(distribution_cfg, "_blend_mode", False):
                from afp.portfolio.blended_allocator import build_blended_portfolio
                target = build_blended_portfolio(
                    d.date(), predictions_with_context, vol_est,
                    portfolio_cfg, distribution_cfg, k=k, sector_map=sector_map)
            elif distribution_cfg is not None:
                from afp.portfolio.distribution_allocator import build_distribution_portfolio
                target = build_distribution_portfolio(
                    d.date(), predictions_with_context, vol_est,
                    portfolio_cfg, distribution_cfg, k=k, sector_map=sector_map)
            else:
                target = build_target_portfolio(d.date(), predictions_with_context, vol_est,
                                                portfolio_cfg, k=k, sector_map=sector_map)
            target_map = dict(zip(target.weights["internal_company_id"], target.weights["weight"]))
            target_cash = target.cash_weight
            # Phase 39: thermodynamic regime filter — exposure scale per day.
            if (regime_filter_cfg is not None and getattr(regime_filter_cfg, "enabled", False)
                    and regime_temperature_series is not None):
                from afp.portfolio.regime_filter import regime_scale_at
                rscale = regime_scale_at(regime_temperature_series, regime_filter_cfg, d.date())
                if rscale != 1.0:
                    target_map = {k_: v * rscale for k_, v in target_map.items()}
                    target_cash = max(0.0, 1.0 - sum(target_map.values()))
            # Phase 44: drawdown-targeting wrapper — scale equity slice by
            # realized DD over the past `lookback_days`. Applied multiplicatively
            # with regime filter and (below) vol target.
            if dd_target_cfg is not None and getattr(dd_target_cfg, "enabled", False):
                from afp.portfolio.dd_target import compute_dd_scale
                if len(daily_rows) > 20:
                    past_ret = pd.Series([r["net_return"] for r in daily_rows])
                    dd_scale = compute_dd_scale(past_ret, dd_target_cfg, dd_target_prev)
                else:
                    dd_scale = 1.0
                dd_target_prev = dd_scale
                if dd_scale != 1.0:
                    target_map = {k_: v * dd_scale for k_, v in target_map.items()}
                    target_cash = max(0.0, 1.0 - sum(target_map.values()))
            # Phase 46: beta-targeting wrapper — scale by inverse rolling
            # beta to SPY. Applied after dd_target, before vol_target.
            if (beta_target_cfg is not None and getattr(beta_target_cfg, "enabled", False)
                    and beta_market_returns is not None):
                from afp.portfolio.beta_target import compute_beta_scale
                if len(daily_rows) > 20:
                    past_ret = pd.Series([r["net_return"] for r in daily_rows],
                                          index=pd.to_datetime([r["date"] for r in daily_rows]))
                    b_scale = compute_beta_scale(past_ret, beta_market_returns,
                                                 beta_target_cfg, beta_target_prev)
                else:
                    b_scale = 1.0
                beta_target_prev = b_scale
                if b_scale != 1.0:
                    target_map = {k_: v * b_scale for k_, v in target_map.items()}
                    target_cash = max(0.0, 1.0 - sum(target_map.values()))
            # Phase 50: vol-of-vol targeting — heteroscedasticity-aware
            # de-risking based on the strategy's own vol-of-vol z-score.
            if vov_target_cfg is not None and getattr(vov_target_cfg, "enabled", False):
                from afp.portfolio.vov_target import compute_vov_scale
                if len(daily_rows) > vov_target_cfg.sigma_window + vov_target_cfg.vov_window:
                    past_ret = pd.Series([r["net_return"] for r in daily_rows])
                    v_scale = compute_vov_scale(past_ret, vov_target_cfg, vov_target_prev)
                else:
                    v_scale = 1.0
                vov_target_prev = v_scale
                if v_scale != 1.0:
                    target_map = {k_: v * v_scale for k_, v in target_map.items()}
                    target_cash = max(0.0, 1.0 - sum(target_map.values()))
            # Phase 60: Gutenberg-Richter foreshock detector — anomalous
            # cluster of small drawdowns precedes large ones (geophysics).
            if gr_cfg is not None and getattr(gr_cfg, "enabled", False):
                from afp.portfolio.gutenberg_richter import compute_gr_scale
                if len(daily_rows) > gr_cfg.history_window + gr_cfg.count_window:
                    past_ret = pd.Series([r["net_return"] for r in daily_rows])
                    g_scale = compute_gr_scale(past_ret, gr_cfg, gr_prev)
                else:
                    g_scale = 1.0
                gr_prev = g_scale
                if g_scale != 1.0:
                    target_map = {k_: v * g_scale for k_, v in target_map.items()}
                    target_cash = max(0.0, 1.0 - sum(target_map.values()))
            # Phase 61: Kuramoto synchronization order parameter — coupled
            # oscillator phase coherence across the universe (Kuramoto 1975).
            if (kuramoto_cfg is not None and getattr(kuramoto_cfg, "enabled", False)
                    and kuramoto_r_series is not None):
                from afp.portfolio.kuramoto import compute_kuramoto_scale
                k_scale = compute_kuramoto_scale(kuramoto_r_series, kuramoto_cfg,
                                                  pd.Timestamp(d), kuramoto_prev)
                kuramoto_prev = k_scale
                if k_scale != 1.0:
                    target_map = {k_: v * k_scale for k_, v in target_map.items()}
                    target_cash = max(0.0, 1.0 - sum(target_map.values()))
            # Phase 32: vol targeting scales the equity slice.
            if vol_target_cfg is not None and getattr(vol_target_cfg, "enabled", False):
                from afp.portfolio.vol_target import compute_portfolio_vol_scale
                if len(daily_rows) > 20:
                    past_ret = pd.Series([r["net_return"] for r in daily_rows])
                    scale = compute_portfolio_vol_scale(past_ret, vol_target_cfg, vol_target_prev)
                else:
                    scale = 1.0
                vol_target_prev = scale
                vol_target_history.append(scale)
                if scale != 1.0:
                    target_map = {k_: v * scale for k_, v in target_map.items()}
                    target_cash = max(0.0, 1.0 - sum(target_map.values()))

            all_ids = set(current_weights) | set(target_map)
            for icid in all_ids:
                old_w = current_weights.get(icid, 0.0)
                new_w = target_map.get(icid, 0.0)
                trade_w = new_w - old_w
                if abs(trade_w) > 1e-9:
                    trades_rows.append({
                        "date": d.date(),
                        "internal_company_id": icid,
                        "old_weight": old_w,
                        "new_weight": new_w,
                        "trade_weight": trade_w,
                        "trade_cost": abs(trade_w) * tcost_rate,
                        "reason": "rebalance" if old_w > 0 and new_w > 0 else
                                 ("new_signal" if new_w > 0 else "exit"),
                    })
                    turnover += abs(trade_w)
                    cost += abs(trade_w) * tcost_rate
            turnover *= 0.5
            current_weights = {k_: v for k_, v in target_map.items() if v > 0}
            cash_weight = target_cash

        net_return = gross_return - cost
        portfolio_value *= (1.0 + net_return)

        daily_rows.append({
            "date": d.date(),
            "gross_return": gross_return,
            "transaction_cost_return": cost,
            "net_return": net_return,
            "portfolio_value": portfolio_value,
            "cash_weight": cash_weight,
            "n_positions": len(current_weights),
            "turnover": turnover,
        })
        for icid, w in current_weights.items():
            holdings_rows.append({"date": d.date(), "internal_company_id": icid, "weight": w})

        prev_date = d

    daily = pd.DataFrame(daily_rows)
    holdings = pd.DataFrame(holdings_rows)
    trades = pd.DataFrame(trades_rows, columns=["date", "internal_company_id", "old_weight",
                                                "new_weight", "trade_weight", "trade_cost", "reason"])
    return BacktestResult(daily=daily, holdings=holdings, trades=trades)
