"""Phase 52 experiment: rank-average ensemble of three models, with RMT+DD stack.

The three models being ensembled:
- LambdaRank (the current best, Phase 29 honest run)
- Ridge + price features (Phase 21)
- Diffusion v2 (Phase 23)

All three already produced predictions parquets on the SAME sample_id
universe. We rank-average (cross-sectionally per entry-date quarter)
their predicted_signal to produce an ensemble signal, then run that
through the existing rmt_minvar + dd_target stack.

Variant 1 — ensemble3 alone
Variant 2 — ensemble3 + dd_target (compare to rmt_minvar + dd_target = 1.084 / -0.286)

Methodology:
- Equal weight (1/3 each), no tuning over weights.
- Cross-sectional rank within each (entry_date quarter) — standard
  bagging form for cross-sectional signals.
- Single-pass evaluation.
"""

from __future__ import annotations

import json
import sys
import time
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
from afp.portfolio.dd_target import DrawdownTargetConfig
from afp.portfolio.risk_models import RiskConfig, VolEstimator
from afp.portfolio.rmt_allocator import build_rmt_target_portfolio


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


def _rmt(d, p, ve, c, k, sec):
    return build_rmt_target_portfolio(d, p, ve, c, k=k, sector_map=sec)


def _build_ensemble_predictions(p_lr: pd.DataFrame, p_ridge: pd.DataFrame,
                                p_diff: pd.DataFrame) -> pd.DataFrame:
    """Rank-average three model predictions on shared sample_ids.

    For each (entry_date, exit_date) quarter group, rank each model's
    predicted_signal cross-sectionally, then average the percentile
    ranks. Recenter to [-1, 1] like a normalized signal.
    """
    base = p_lr.set_index("sample_id")[
        ["predicted_signal", "internal_company_id", "entry_date", "exit_date",
         "ex_ante_scale", "split", "model_id", "model_type", "prediction_created_at",
         "target_normalized_signal", "raw_log_return"]].copy()
    base["lr_signal"] = base.pop("predicted_signal")

    r = p_ridge.set_index("sample_id")["predicted_signal"].rename("ridge_signal")
    d = p_diff.set_index("sample_id")["predicted_signal"].rename("diff_signal")
    df = base.join(r).join(d)
    df = df.dropna(subset=["lr_signal", "ridge_signal", "diff_signal"])
    print(f"ensemble samples after join: {len(df)}", flush=True)

    # rank-average cross-sectionally per entry_date quarter
    quarter = pd.to_datetime(df["entry_date"]).dt.to_period("Q").astype(str)
    out = []
    for q, grp in df.groupby(quarter):
        ranks = pd.DataFrame({
            "lr": grp["lr_signal"].rank(pct=True),
            "ridge": grp["ridge_signal"].rank(pct=True),
            "diff": grp["diff_signal"].rank(pct=True),
        }, index=grp.index)
        avg_rank = ranks.mean(axis=1)
        # Map [0, 1] back to [-1, 1]
        ens_signal = (avg_rank * 2.0 - 1.0)
        sub = grp.copy()
        sub["predicted_signal"] = ens_signal
        out.append(sub)
    result = pd.concat(out).reset_index()
    result["model_id"] = "ensemble3_lambdarank_ridge_diff_v2"
    result["model_type"] = "rank_avg_ensemble"
    # Reorder cols to match production schema
    cols_in_order = ["sample_id", "predicted_signal", "split", "model_id",
                     "model_type", "prediction_created_at", "internal_company_id",
                     "entry_date", "exit_date", "ex_ante_scale",
                     "target_normalized_signal", "raw_log_return"]
    return result[cols_in_order]


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
    config_path = "configs/deployment_v1000.yaml"
    cfg = resolve_config(config_path)
    bt_cfg, pcfg, rcfg = _build_cfgs(cfg)
    k = cfg["portfolio"]["signal"]["k"]

    p_lr = pd.read_parquet("artifacts/predictions/lambdarank_v3_p100_s42.parquet")
    p_ridge = pd.read_parquet("artifacts/predictions/ridge_price_v1000.parquet")
    p_diff = pd.read_parquet("artifacts/predictions/diffusion_v2.parquet")
    print(f"LR={len(p_lr)}  Ridge={len(p_ridge)}  Diff={len(p_diff)}", flush=True)

    ensemble = _build_ensemble_predictions(p_lr, p_ridge, p_diff)
    out_pred_path = Path("artifacts/predictions/ens3_v2_phys.parquet")
    ensemble.to_parquet(out_pred_path)
    print(f"saved ensemble predictions to {out_pred_path}", flush=True)

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
    results.append(_run("ensemble3_rmt", ensemble, prices, benchmarks, calendar,
                        bt_cfg, pcfg, rcfg, k, _rmt, shared_vol))
    results.append(_run("ensemble3_rmt+dd_target", ensemble, prices, benchmarks, calendar,
                        bt_cfg, pcfg, rcfg, k, _rmt, shared_vol, dd_cfg=dd_cfg))

    out_dir = Path("artifacts/experiments/phys_v10")
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
