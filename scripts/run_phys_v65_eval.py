"""Phase 65 backtest: Phase 62 cross-features model evaluated on tier2 + tier3.

Confirms whether the +0.05 Sharpe / +5% Calmar improvement from
adding cross-disciplinary features in the model generalizes to two
external universes the model was never trained on.
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
from afp.portfolio.risk_models import RiskConfig, VolEstimator
from afp.portfolio.rmt_barbell_allocator import build_rmt_barbell_target_portfolio


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


def _rb(d, p, ve, c, k, sec):
    return build_rmt_barbell_target_portfolio(d, p, ve, c, k=k, sector_map=sec)


def _run(name, predictions, prices, benchmarks, calendar, bt_cfg, pcfg, rcfg, k,
         vol_estimator, dd_cfg):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    result = run_backtest(predictions, prices, calendar, bt_cfg, pcfg, rcfg,
                          k=k, sector_map=None, custom_allocator=_rb,
                          vol_estimator=vol_estimator, dd_target_cfg=dd_cfg)
    print(f"  run_backtest: {time.time()-t0:.1f}s", flush=True)
    report = build_report(result, prices, benchmarks, predictions=predictions,
                          target_positions=pcfg.target_positions)
    s = report["strategy"]
    print(f"  Sharpe={s['sharpe']:.3f}  Sortino={s['sortino']:.3f}  "
          f"MDD={s['max_drawdown']:.3f}  Calmar={s['calmar']:.3f}  "
          f"AnnRet={s['annualized_return']:.3f}  AnnVol={s['annualized_volatility']:.3f}",
          flush=True)
    return {"variant": name, "sharpe": s["sharpe"], "sortino": s["sortino"],
            "max_drawdown": s["max_drawdown"], "calmar": s["calmar"],
            "annualized_return": s["annualized_return"],
            "annualized_volatility": s["annualized_volatility"]}


def main():
    config_path = "configs/deployment_v1000.yaml"
    cfg = resolve_config(config_path)
    bt_cfg, pcfg, rcfg = _build_cfgs(cfg)
    k = cfg["portfolio"]["signal"]["k"]
    benchmarks = pd.read_parquet("data/processed/benchmarks_daily.parquet")
    cal_df = pd.read_parquet("data/processed/trading_calendar.parquet")
    calendar = TradingCalendar.from_dates(cal_df["date"].tolist())
    dd_cfg = DrawdownTargetConfig(enabled=True, lookback_days=252,
                                   dd_trigger=0.10, alpha=1.0,
                                   scale_floor=0.30, blend=0.5)

    runs = [
        ("tier2_cross+rmt_barbell+dd",
         "artifacts/predictions/lambdarank_tier2_cross.parquet",
         "data/processed/tier2/prices_daily.parquet"),
        ("tier3_cross+rmt_barbell+dd",
         "artifacts/predictions/lambdarank_tier3_cross.parquet",
         "data/processed/tier3/prices_daily.parquet"),
    ]

    results = []
    for name, pred_path, prices_path in runs:
        predictions = pd.read_parquet(pred_path)
        prices = pd.read_parquet(prices_path)
        t0 = time.time()
        shared_vol = VolEstimator(prices, rcfg)
        print(f"vol estimator for {name}: {time.time()-t0:.1f}s", flush=True)
        results.append(_run(name, predictions, prices, benchmarks, calendar,
                             bt_cfg, pcfg, rcfg, k, shared_vol, dd_cfg))

    out_dir = Path("artifacts/experiments/phys_v65_crossfeat_universe")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(results).to_csv(out_dir / "metrics.csv", index=False)

    print("\n=== Phase 65: Phase 62 model on tier2 + tier3 ===", flush=True)
    print("  Phase 62 (tier1 s42):       Sharpe=1.128  MDD=-0.278  Calmar=0.917", flush=True)
    for r in results:
        print(f"{r['variant']:35s}  Sharpe={r['sharpe']:.3f}  "
              f"MDD={r['max_drawdown']:.3f}  Calmar={r['calmar']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
