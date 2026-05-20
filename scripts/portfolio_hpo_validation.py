"""Phase 33: portfolio HPO on the VALIDATION window only.

The earlier "concentration sweep" (N=10..100) was implicitly run on the test
window — that's data snooping. This script runs the exact same sweep on the
2016-2019 validation window (using the same LambdaRank predictions parquet,
which already includes val + test predictions), picks the highest validation
Sharpe config, and reports it. The final test number is then a single,
non-snooped backtest with the winning config.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afp.backtest.engine import BacktestConfig, run_backtest
from afp.backtest.metrics import summary as backtest_summary
from afp.data.trading_calendar import TradingCalendar
from afp.portfolio.allocation import PortfolioConfig
from afp.portfolio.risk_models import RiskConfig


def _backtest_window(predictions, prices, calendar, start, end,
                     target_positions, max_single, min_position_weight,
                     sector_map=None, max_sector_weight=None):
    pcfg = PortfolioConfig(
        min_positive_signal=0.0,
        signal_power=1.0,
        min_positions=5,
        target_positions=target_positions,
        max_positions=int(target_positions * 1.3),
        max_single_stock_weight=max_single,
        min_position_weight=min_position_weight,
        allow_cash=True,
        leverage=1.0,
        max_sector_weight=max_sector_weight,
    )
    rcfg = RiskConfig(lookback_days=252, min_daily_vol=0.005, max_daily_vol=0.10)
    bt_cfg = BacktestConfig(start_date=start, end_date=end,
                            transaction_cost_bps_per_trade=10.0)
    result = run_backtest(predictions, prices, calendar, bt_cfg, pcfg, rcfg,
                          k=2.5, sector_map=sector_map)
    daily_returns = result.daily.set_index(pd.to_datetime(result.daily["date"]))["net_return"]
    turnover = result.daily.set_index(pd.to_datetime(result.daily["date"]))["turnover"]
    return backtest_summary(daily_returns, turnover, 252)


def main():
    predictions = pd.read_parquet("artifacts/predictions/lambdarank_v3_p25_s42.parquet")
    prices = pd.read_parquet("data/processed/prices_daily.parquet")
    cal_df = pd.read_parquet("data/processed/trading_calendar.parquet")
    calendar = TradingCalendar.from_dates(cal_df["date"].tolist())

    sectors_path = Path("data/processed/sectors.parquet")
    sector_map = None
    if sectors_path.exists():
        sec_df = pd.read_parquet(sectors_path)
        sector_map = dict(zip(sec_df["internal_company_id"], sec_df["sic_division"]))

    # Sweep restricted to N ≥ 50 — user flagged concentration as overfit risk.
    # Phases 30-32's N=10..40 sweep produced 1.04-1.15 but was test-snooped.
    configs = [
        # (N, max_single, min_w, max_sector_weight)
        (50,  0.05, 0.0025, None),       # RFC default
        (50,  0.04, 0.002,  0.25),       # default + sector cap
        (50,  0.05, 0.0025, 0.30),       # default + looser sector cap
        (75,  0.03, 0.001,  None),       # diversified
        (75,  0.03, 0.001,  0.22),       # diversified + sector cap
        (100, 0.03, 0.001,  None),       # wider
        (100, 0.02, 0.001,  0.20),       # wider + sector cap
        (150, 0.02, 0.001,  None),       # very wide
    ]

    print(f"\n{'config':<35} {'Val Sharpe':>10}  {'Val Ret':>8}  {'Val MDD':>8}")
    print("-" * 70)
    val_results = []
    for tp, ms, mw, msw in configs:
        cfg_name = f"N={tp:>3d} ms={ms:.2f} mw={mw:.3f} sec={msw if msw else '—'}"
        m = _backtest_window(predictions, prices, calendar, "2016-01-04", "2019-12-31",
                             tp, ms, mw, sector_map=sector_map if msw else None,
                             max_sector_weight=msw)
        val_results.append((cfg_name, tp, ms, mw, msw, m))
        print(f"  {cfg_name:<35} {m['sharpe']:>10.2f}  {m['annualized_return']*100:>7.1f}%  {m['max_drawdown']*100:>7.1f}%")

    # Pick the config with the best validation Sharpe.
    val_results.sort(key=lambda x: x[5]["sharpe"], reverse=True)
    best = val_results[0]
    print(f"\n>>> Winner on val: {best[0]} -> Sharpe {best[5]['sharpe']:.2f}")

    # ONE-SHOT test backtest with the winning config.
    print("\n>>> Final test backtest (single shot, no further tuning):")
    test_metrics = _backtest_window(predictions, prices, calendar, "2020-01-02", None,
                                    best[1], best[2], best[3],
                                    sector_map=sector_map if best[4] else None,
                                    max_sector_weight=best[4])
    print(f"  Test Sharpe        {test_metrics['sharpe']:.3f}")
    print(f"  Test Ann.Return    {test_metrics['annualized_return']*100:.1f}%")
    print(f"  Test Ann.Vol       {test_metrics['annualized_volatility']*100:.1f}%")
    print(f"  Test MDD           {test_metrics['max_drawdown']*100:.1f}%")
    print(f"  Test Sortino       {test_metrics['sortino']:.2f}")
    print(f"  Test Calmar        {test_metrics['calmar']:.2f}")


if __name__ == "__main__":
    main()
