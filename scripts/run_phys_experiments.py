"""Phase 38-40: physics-inspired allocator experiments.

Single-pass evaluation: take the FROZEN best predictions
(`lambdarank_v3_p100_s42.parquet`) — no model retraining — and compare:

  baseline   : current inverse-vol allocator (Phase 26 best)
  rmt_minvar : RMT-cleaned signal-tilted min-variance (Phase 38)
  max_ent    : Boltzmann/softmax allocator (Phase 40)
  baseline + regime : baseline scaled by thermodynamic regime (Phase 39)
  rmt_minvar + regime
  max_ent + regime
  max_ent + regime, low-vol blend (inverse_vol_blend=1.0)

All hyperparameters are theoretical (see module docstrings). No knob
is tuned on val or test data.

Outputs metrics to `artifacts/experiments/phys_v1/metrics.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afp.backtest.engine import BacktestConfig, run_backtest
from afp.backtest.report import build_report
from afp.cli._common import resolve_config
from afp.data.trading_calendar import TradingCalendar
from afp.portfolio.allocation import PortfolioConfig
from afp.portfolio.max_entropy_allocator import build_max_entropy_target_portfolio
from afp.portfolio.regime_filter import (
    RegimeFilterConfig,
    cross_sectional_temperature,
)
from afp.portfolio.risk_models import RiskConfig
from afp.portfolio.rmt_allocator import build_rmt_target_portfolio


def _build_cfgs(cfg_dict, use_sector_caps=False):
    bt_cfg = BacktestConfig(
        start_date=cfg_dict["backtest"]["start_date"],
        end_date=cfg_dict["backtest"]["end_date"],
        initial_value=cfg_dict["backtest"].get("initial_value", 1.0),
        transaction_cost_bps_per_trade=cfg_dict["portfolio"]["costs"][
            "transaction_cost_bps_per_trade"],
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
        max_sector_weight=cfg_dict["portfolio"]["sector"].get("max_sector_weight")
        if use_sector_caps else None,
    )
    rcfg = RiskConfig(
        lookback_days=cfg_dict["portfolio"]["risk"]["risk_lookback_days"],
        min_daily_vol=cfg_dict["portfolio"]["risk"]["min_daily_vol"],
        max_daily_vol=cfg_dict["portfolio"]["risk"]["max_daily_vol"],
        covariance_shrinkage_lambda=cfg_dict["portfolio"]["risk"]["covariance_shrinkage_lambda"],
    )
    return bt_cfg, pcfg, rcfg


def _rmt_allocator(d, preds, vol_est, pcfg, k, sec_map):
    return build_rmt_target_portfolio(d, preds, vol_est, pcfg, k=k, sector_map=sec_map)


def _max_entropy_allocator(d, preds, vol_est, pcfg, k, sec_map):
    return build_max_entropy_target_portfolio(d, preds, vol_est, pcfg, k=k, sector_map=sec_map,
                                              inverse_vol_blend=0.5)


def _max_entropy_lowvol_allocator(d, preds, vol_est, pcfg, k, sec_map):
    return build_max_entropy_target_portfolio(d, preds, vol_est, pcfg, k=k, sector_map=sec_map,
                                              inverse_vol_blend=1.0)


def _run_variant(name, predictions, prices, benchmarks, calendar, bt_cfg, pcfg, rcfg, k,
                 custom_allocator=None, regime_cfg=None, regime_temp=None,
                 vol_estimator=None):
    import time
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    result = run_backtest(predictions, prices, calendar, bt_cfg, pcfg, rcfg,
                          k=k, sector_map=None,
                          custom_allocator=custom_allocator,
                          regime_filter_cfg=regime_cfg,
                          regime_temperature_series=regime_temp,
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
          f"AnnRet={s['annualized_return']:.3f}  AnnVol={s['annualized_volatility']:.3f}")
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
    bt_cfg, pcfg, rcfg = _build_cfgs(cfg, use_sector_caps=False)
    k = cfg["portfolio"]["signal"]["k"]

    predictions = pd.read_parquet(pred_path)
    prices = pd.read_parquet("data/processed/prices_daily.parquet")
    benchmarks = pd.read_parquet("data/processed/benchmarks_daily.parquet")
    cal_df = pd.read_parquet("data/processed/trading_calendar.parquet")
    calendar = TradingCalendar.from_dates(cal_df["date"].tolist())

    # Precompute the cross-sectional temperature panel from prices.
    price_pivot = (prices.assign(date=pd.to_datetime(prices["date"]))
                         .pivot_table(index="date", columns="internal_company_id",
                                      values="adjusted_close", aggfunc="last")
                         .ffill())
    T_series = cross_sectional_temperature(price_pivot)
    regime_cfg = RegimeFilterConfig(enabled=True, lookback_days=252, boltzmann_kt=2.0,
                                    scale_floor=0.30, scale_ceiling=1.0, smoothing=0.5)

    print(f"loaded predictions shape={predictions.shape}, dates "
          f"{predictions['entry_date'].min()} → {predictions['entry_date'].max()}", flush=True)
    print(f"backtest window: {bt_cfg.start_date} → {bt_cfg.end_date}", flush=True)

    # Build VolEstimator ONCE and share across all variants.
    print("building shared VolEstimator...", flush=True)
    import time
    from afp.portfolio.risk_models import VolEstimator
    t0 = time.time()
    shared_vol = VolEstimator(prices, rcfg)
    print(f"  vol estimator: {time.time()-t0:.1f}s", flush=True)

    results = []
    results.append(_run_variant("baseline_inverse_vol",
                                predictions, prices, benchmarks, calendar,
                                bt_cfg, pcfg, rcfg, k, vol_estimator=shared_vol))
    results.append(_run_variant("rmt_minvar",
                                predictions, prices, benchmarks, calendar,
                                bt_cfg, pcfg, rcfg, k,
                                custom_allocator=_rmt_allocator,
                                vol_estimator=shared_vol))
    results.append(_run_variant("max_entropy",
                                predictions, prices, benchmarks, calendar,
                                bt_cfg, pcfg, rcfg, k,
                                custom_allocator=_max_entropy_allocator,
                                vol_estimator=shared_vol))
    results.append(_run_variant("max_entropy_lowvol_blend",
                                predictions, prices, benchmarks, calendar,
                                bt_cfg, pcfg, rcfg, k,
                                custom_allocator=_max_entropy_lowvol_allocator,
                                vol_estimator=shared_vol))
    results.append(_run_variant("baseline_inverse_vol+regime",
                                predictions, prices, benchmarks, calendar,
                                bt_cfg, pcfg, rcfg, k,
                                regime_cfg=regime_cfg, regime_temp=T_series,
                                vol_estimator=shared_vol))
    results.append(_run_variant("rmt_minvar+regime",
                                predictions, prices, benchmarks, calendar,
                                bt_cfg, pcfg, rcfg, k,
                                custom_allocator=_rmt_allocator,
                                regime_cfg=regime_cfg, regime_temp=T_series,
                                vol_estimator=shared_vol))
    results.append(_run_variant("max_entropy+regime",
                                predictions, prices, benchmarks, calendar,
                                bt_cfg, pcfg, rcfg, k,
                                custom_allocator=_max_entropy_allocator,
                                regime_cfg=regime_cfg, regime_temp=T_series,
                                vol_estimator=shared_vol))
    results.append(_run_variant("max_entropy_lowvol+regime",
                                predictions, prices, benchmarks, calendar,
                                bt_cfg, pcfg, rcfg, k,
                                custom_allocator=_max_entropy_lowvol_allocator,
                                regime_cfg=regime_cfg, regime_temp=T_series,
                                vol_estimator=shared_vol))

    out_dir = Path("artifacts/experiments/phys_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(results).to_csv(out_dir / "metrics.csv", index=False)

    print("\n=== LEADERBOARD (sorted by Sharpe) ===")
    sorted_r = sorted(results, key=lambda r: r["sharpe"], reverse=True)
    for r in sorted_r:
        print(f"{r['variant']:35s}  Sharpe={r['sharpe']:.3f}  "
              f"MDD={r['max_drawdown']:.3f}  Calmar={r['calmar']:.3f}")


if __name__ == "__main__":
    main()
