"""Phase 51 experiment: allocator-level ensemble (inverse-vol + RMT min-var).

Variants:
- ensemble (alone)
- ensemble + dd_target (with the Phase 44 wrapper)

2 variants, ~12 min.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afp.backtest.engine import BacktestConfig, run_backtest
from afp.backtest.report import build_report
from afp.cli._common import resolve_config
from afp.data.trading_calendar import TradingCalendar
from afp.portfolio.allocation import PortfolioConfig
from afp.portfolio.dd_target import DrawdownTargetConfig
from afp.portfolio.ensemble_allocator import build_ensemble_target_portfolio
from afp.portfolio.risk_models import RiskConfig, VolEstimator


def _build_cfgs(cfg_dict):
    bt_cfg = BacktestConfig(
        start_date=cfg_dict["backtest"]["start_date"],
        end_date=cfg_dict["backtest"]["end_date"],
        initial_value=cfg_dict["backtest"].get("initial_value", 1.0),
        transaction_cost_bps_per_trade=cfg_dict["portfolio"]["costs"]["transaction_cost_bps_per_trade"],
        cash_return_daily=cfg_dict["portfolio"]["costs"].get("cash_return", 0.0),
    )
    pcfg = PortfolioConfig(
        min_positive_signal=cfg_dict["portfolio"]["signal"]["min_positive_signal"],
        signal_power=cfg_dict["portfolio"]["signal"]["signal_power"],
        min_positions=cfg_dict["portfolio"]["positions"]["min_positions"],
        target_positions=cfg_dict["portfolio"]["positions"]["target_positions"],
        max_positions=cfg_dict["portfolio"]["positions"]["max_positions"],
        max_single_stock_weight=cfg_dict["portfolio"]["positions"]["max_single_stock_weight"],
        min_position_weight=cfg_dict["portfolio"]["positions"]["min_position_weight"],
        allow_cash=cfg_dict["portfolio"]["allow_cash"],
        leverage=cfg_dict["portfolio"]["leverage"],
    )
    rcfg = RiskConfig(
        lookback_days=cfg_dict["portfolio"]["risk"]["risk_lookback_days"],
        min_daily_vol=cfg_dict["portfolio"]["risk"]["min_daily_vol"],
        max_daily_vol=cfg_dict["portfolio"]["risk"]["max_daily_vol"],
        covariance_shrinkage_lambda=cfg_dict["portfolio"]["risk"]["covariance_shrinkage_lambda"],
    )
    return bt_cfg, pcfg, rcfg


def _ens(d, p, ve, c, k, sec):
    return build_ensemble_target_portfolio(d, p, ve, c, k=k, sector_map=sec)


def _run(name, predictions, prices, benchmarks, calendar, bt_cfg, pcfg, rcfg, k,
         custom_allocator, vol_estimator, dd_cfg=None):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    result = run_backtest(predictions, prices, calendar, bt_cfg, pcfg, rcfg,
                          k=k, sector_map=None,
                          custom_allocator=custom_allocator,
                          vol_estimator=vol_estimator,
                          dd_target_cfg=dd_cfg)
    print(f"  run_backtest: {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    report = build_report(result, prices, benchmarks,
                          predictions=predictions,
                          target_positions=pcfg.target_positions)
    print(f"  build_report: {time.time()-t0:.1f}s", flush=True)
    s = report["strategy"]
    print(f"  Sharpe={s['sharpe']:.3f}  Sortino={s['sortino']:.3f}  "
          f"MDD={s['max_drawdown']:.3f}  Calmar={s['calmar']:.3f}  "
          f"AnnRet={s['annualized_return']:.3f}  AnnVol={s['annualized_volatility']:.3f}",
          flush=True)
    return {
        "variant": name,
        "sharpe": s["sharpe"],
        "sortino": s["sortino"],
        "max_drawdown": s["max_drawdown"],
        "calmar": s["calmar"],
        "annualized_return": s["annualized_return"],
        "annualized_volatility": s["annualized_volatility"],
        "annualized_turnover": s.get("annualized_turnover", float("nan")),
        "n_days": int(s["n_days"]),
    }


def main():
    pred_path = "artifacts/predictions/lambdarank_v3_p100_s42.parquet"
    config_path = "configs/deployment_v1000.yaml"
    cfg = resolve_config(config_path)
    bt_cfg, pcfg, rcfg = _build_cfgs(cfg)
    k = cfg["portfolio"]["signal"]["k"]

    predictions = pd.read_parquet(pred_path)
    prices = pd.read_parquet("data/processed/prices_daily.parquet")
    benchmarks = pd.read_parquet("data/processed/benchmarks_daily.parquet")
    cal_df = pd.read_parquet("data/processed/trading_calendar.parquet")
    calendar = TradingCalendar.from_dates(cal_df["date"].tolist())

    print("building shared VolEstimator...", flush=True)
    t0 = time.time()
    shared_vol = VolEstimator(prices, rcfg)
    print(f"  vol estimator: {time.time()-t0:.1f}s", flush=True)

    dd_cfg = DrawdownTargetConfig(enabled=True, lookback_days=252,
                                   dd_trigger=0.10, alpha=1.0,
                                   scale_floor=0.30, blend=0.5)

    results = []
    results.append(_run("ensemble_allocator", predictions, prices, benchmarks, calendar,
                        bt_cfg, pcfg, rcfg, k, _ens, shared_vol))
    results.append(_run("ensemble+dd_target", predictions, prices, benchmarks, calendar,
                        bt_cfg, pcfg, rcfg, k, _ens, shared_vol, dd_cfg=dd_cfg))

    out_dir = Path("artifacts/experiments/phys_v9")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(results).to_csv(out_dir / "metrics.csv", index=False)

    print("\n=== LEADERBOARD (sorted by Calmar) ===", flush=True)
    for r in sorted(results, key=lambda x: x["calmar"], reverse=True):
        print(f"{r['variant']:40s}  Sharpe={r['sharpe']:.3f}  "
              f"MDD={r['max_drawdown']:.3f}  Calmar={r['calmar']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
