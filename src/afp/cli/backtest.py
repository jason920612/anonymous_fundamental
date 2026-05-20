"""`afp backtest` — run the event-driven backtest using a predictions parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from afp.backtest.engine import BacktestConfig, run_backtest
from afp.backtest.report import build_report, write_report
from afp.cli._common import log, require_parquet, resolve_config, write_parquet
from afp.data.trading_calendar import TradingCalendar
from afp.portfolio.allocation import PortfolioConfig
from afp.portfolio.risk_models import RiskConfig


def add_subparser(sub):
    p = sub.add_parser("backtest", help="Backtest a predictions parquet")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--predictions", required=True)
    p.add_argument("--prices", default="data/processed/prices_daily.parquet")
    p.add_argument("--benchmarks", default="data/processed/benchmarks_daily.parquet")
    p.add_argument("--calendar", default="data/processed/trading_calendar.parquet")
    p.add_argument("--sectors", default="data/processed/sectors.parquet",
                   help="Optional sectors parquet (built by `afp build-sectors`). "
                        "If missing or sector caps disabled, sector caps are skipped.")
    p.add_argument("--out-dir", default="reports/backtest")
    p.add_argument("--portfolio-id", default=None)
    p.add_argument("--allocator", choices=["inverse_vol", "distribution", "blended"],
                   default="inverse_vol",
                   help="Phase 24/27: 'distribution' = confidence/std weighting; "
                        "'blended' = equal-risk-active over ALL active names with "
                        "multiplicative tilt by predicted probability_positive")
    p.add_argument("--min-probability-positive", type=float, default=0.55)
    p.add_argument("--blend-with-signal", type=float, default=0.0)
    p.add_argument("--tilt-alpha", type=float, default=0.5,
                   help="Blended allocator: multiplicative tilt strength (0 = pure equal-risk)")
    p.add_argument("--require-positive-signal", action="store_true",
                   help="Blended allocator: drop names with non-positive predicted_signal")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config)
    predictions = require_parquet(args.predictions)
    prices = require_parquet(args.prices)
    benchmarks = require_parquet(args.benchmarks)
    cal_df = require_parquet(args.calendar)
    calendar = TradingCalendar.from_dates(cal_df["date"].tolist())

    bt_cfg = BacktestConfig(
        start_date=cfg["backtest"]["start_date"],
        end_date=cfg["backtest"]["end_date"],
        initial_value=cfg["backtest"].get("initial_value", 1.0),
        transaction_cost_bps_per_trade=cfg["portfolio"]["costs"]["transaction_cost_bps_per_trade"],
        cash_return_daily=cfg["portfolio"]["costs"].get("cash_return", 0.0),
    )
    use_sector_caps = cfg["portfolio"]["sector"].get("use_sector_caps", False)
    pcfg = PortfolioConfig(
        min_positive_signal=cfg["portfolio"]["signal"]["min_positive_signal"],
        signal_power=cfg["portfolio"]["signal"]["signal_power"],
        min_positions=cfg["portfolio"]["positions"]["min_positions"],
        target_positions=cfg["portfolio"]["positions"]["target_positions"],
        max_positions=cfg["portfolio"]["positions"]["max_positions"],
        max_single_stock_weight=cfg["portfolio"]["positions"]["max_single_stock_weight"],
        min_position_weight=cfg["portfolio"]["positions"]["min_position_weight"],
        allow_cash=cfg["portfolio"]["allow_cash"],
        leverage=cfg["portfolio"]["leverage"],
        max_sector_weight=cfg["portfolio"]["sector"].get("max_sector_weight") if use_sector_caps else None,
    )
    rcfg = RiskConfig(
        lookback_days=cfg["portfolio"]["risk"]["risk_lookback_days"],
        min_daily_vol=cfg["portfolio"]["risk"]["min_daily_vol"],
        max_daily_vol=cfg["portfolio"]["risk"]["max_daily_vol"],
        covariance_shrinkage_lambda=cfg["portfolio"]["risk"]["covariance_shrinkage_lambda"],
    )

    # Phase 16: optional sector map for sector caps
    sector_map = None
    sectors_path = Path(args.sectors)
    if sectors_path.exists() and pcfg.max_sector_weight is not None:
        sec_df = require_parquet(sectors_path)
        sector_map = dict(zip(sec_df["internal_company_id"], sec_df["sic_division"]))
        log.info("backtest_sectors_loaded",
                 extra={"n_companies": len(sector_map),
                        "max_sector_weight": pcfg.max_sector_weight})

    distribution_cfg = None
    if args.allocator == "distribution":
        from afp.portfolio.distribution_allocator import DistributionAllocatorConfig
        distribution_cfg = DistributionAllocatorConfig(
            min_probability_positive=args.min_probability_positive,
            blend_with_signal=args.blend_with_signal,
        )
        if "probability_positive" not in predictions.columns:
            log.warning("backtest_distribution_columns_missing",
                        extra={"hint": "predictions parquet lacks probability_positive; "
                                       "falling back to inverse_vol allocator"})
            distribution_cfg = None
    elif args.allocator == "blended":
        from afp.portfolio.blended_allocator import BlendedAllocatorConfig
        distribution_cfg = BlendedAllocatorConfig(
            tilt_alpha=args.tilt_alpha,
            use_probability_positive="probability_positive" in predictions.columns,
            require_positive_signal=args.require_positive_signal,
        )
        # Mark for the engine to dispatch to the blended path.
        setattr(distribution_cfg, "_blend_mode", True)

    log.info("backtest_start", extra={"start": bt_cfg.start_date, "end": bt_cfg.end_date,
                                      "allocator": "distribution" if distribution_cfg else "inverse_vol"})
    result = run_backtest(predictions, prices, calendar, bt_cfg, pcfg, rcfg,
                          k=cfg["portfolio"]["signal"]["k"],
                          sector_map=sector_map,
                          distribution_cfg=distribution_cfg)

    portfolio_id = args.portfolio_id or Path(args.predictions).stem
    out = Path(args.out_dir) / portfolio_id
    out.mkdir(parents=True, exist_ok=True)
    write_parquet(result.daily, out / "daily.parquet")
    write_parquet(result.holdings, out / "holdings.parquet")
    write_parquet(result.trades, out / "trades.parquet")
    report = build_report(result, prices, benchmarks, predictions=predictions,
                          target_positions=pcfg.target_positions)
    write_report(report, out)

    # Phase 11: portfolio attribution
    from afp.backtest.attribution import build_attribution_report
    attribution = build_attribution_report(result.holdings, result.daily, prices,
                                           tcost_bps=bt_cfg.transaction_cost_bps_per_trade)
    (out / "attribution.json").write_text(json.dumps(attribution, indent=2, default=str))

    log.info("backtest_done", extra={"portfolio_id": portfolio_id, **report["strategy"]})
    return 0
