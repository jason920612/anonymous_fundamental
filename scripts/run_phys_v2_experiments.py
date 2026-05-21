"""Phase 42-43: HRP + tail-aware allocator evaluation.

Follow-up to Phase 41. Findings from Phase 41:
- RMT min-var (Phase 38) → +0.08 Sharpe (1.018 → 1.096). KEEPER.
- Regime filter (Phase 39) → HURTS (-0.035 Sharpe). DROP.
- Max-entropy (Phase 40) → too diversified (-0.19). DROP.

This script evaluates two new candidates *that focus on MDD*:

- hrp_rmt: Hierarchical Risk Parity on the RMT-cleaned covariance.
  Composes orthogonally with Phase 38: RMT denoises, HRP avoids
  inversion entirely.

- tail_aware: weight by signal / Expected-Shortfall instead of
  signal / vol. ES penalizes heavy left-tails which σ ignores.

- rmt_minvar_blend: rmt_minvar with risk_aversion = 2.0 (more
  tilt toward signal) and 10.0 (more toward min-var). Lock both
  *theoretically* at γ=5 was the default, but we never tested
  sensitivity. The exam: is the previous γ=5 setting near-optimal
  by accident, or is the structure robust? This is NOT tuning —
  we only run it once and check.
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
from afp.portfolio.hrp_allocator import build_hrp_target_portfolio
from afp.portfolio.risk_models import RiskConfig, VolEstimator
from afp.portfolio.rmt_allocator import build_rmt_target_portfolio
from afp.portfolio.tail_aware_allocator import build_tail_aware_target_portfolio


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


def _hrp_alloc(d, preds, vol_est, pcfg, k, sec):
    return build_hrp_target_portfolio(d, preds, vol_est, pcfg, k=k, sector_map=sec)


def _tail_alloc(d, preds, vol_est, pcfg, k, sec):
    return build_tail_aware_target_portfolio(d, preds, vol_est, pcfg, k=k, sector_map=sec)


def _rmt_alloc_lo(d, preds, vol_est, pcfg, k, sec):
    return build_rmt_target_portfolio(d, preds, vol_est, pcfg, k=k, sector_map=sec,
                                      risk_aversion=2.0)


def _rmt_alloc_hi(d, preds, vol_est, pcfg, k, sec):
    return build_rmt_target_portfolio(d, preds, vol_est, pcfg, k=k, sector_map=sec,
                                      risk_aversion=10.0)


def _run(name, predictions, prices, benchmarks, calendar, bt_cfg, pcfg, rcfg, k,
         custom_allocator, vol_estimator):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    result = run_backtest(predictions, prices, calendar, bt_cfg, pcfg, rcfg,
                          k=k, sector_map=None,
                          custom_allocator=custom_allocator,
                          vol_estimator=vol_estimator)
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

    results = []
    results.append(_run("hrp_rmt", predictions, prices, benchmarks, calendar,
                        bt_cfg, pcfg, rcfg, k, _hrp_alloc, shared_vol))
    results.append(_run("tail_aware", predictions, prices, benchmarks, calendar,
                        bt_cfg, pcfg, rcfg, k, _tail_alloc, shared_vol))
    results.append(_run("rmt_minvar_low_gamma", predictions, prices, benchmarks, calendar,
                        bt_cfg, pcfg, rcfg, k, _rmt_alloc_lo, shared_vol))
    results.append(_run("rmt_minvar_high_gamma", predictions, prices, benchmarks, calendar,
                        bt_cfg, pcfg, rcfg, k, _rmt_alloc_hi, shared_vol))

    out_dir = Path("artifacts/experiments/phys_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(results).to_csv(out_dir / "metrics.csv", index=False)

    print("\n=== LEADERBOARD (sorted by Sharpe) ===", flush=True)
    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        print(f"{r['variant']:30s}  Sharpe={r['sharpe']:.3f}  "
              f"MDD={r['max_drawdown']:.3f}  Calmar={r['calmar']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
