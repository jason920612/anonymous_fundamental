"""Phase 8 tests — accounting, drawdown, turnover, end-to-end smoke."""

import numpy as np
import pandas as pd
import pytest

from afp.backtest.engine import BacktestConfig, run_backtest
from afp.backtest.metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
    summary,
)
from afp.backtest.report import build_report
from afp.features.encoder import AnonymousFeatureEncoder
from afp.features.sample_builder import build_event_samples
from afp.models.train import attach_prediction_context, train_and_predict
from afp.portfolio.allocation import PortfolioConfig
from afp.portfolio.risk_models import RiskConfig
from afp.targets.target_transform import TargetConfig, apply as apply_target
from afp.targets.volatility_scale import ScaleConfig, compute_ex_ante_scale_for_samples
from tests.fixtures import make_synthetic_world


def test_annualized_return_basic():
    s = pd.Series([0.01, -0.01, 0.02, 0.01])
    n = len(s)
    expected = float((1 + s).prod()) ** (252 / n) - 1
    assert annualized_return(s) == pytest.approx(expected, rel=1e-9)


def test_max_drawdown_known_path():
    s = pd.Series([0.1, -0.1, 0.0, -0.5, 0.5])
    cum = (1 + s).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1).min()
    assert max_drawdown(s) == pytest.approx(dd)


def test_sharpe_zero_when_vol_zero():
    s = pd.Series([0.0] * 50)
    assert sharpe_ratio(s) == 0.0


def test_summary_keys_present():
    s = pd.Series(np.random.RandomState(0).normal(0.001, 0.01, 200))
    out = summary(s, pd.Series([0.05] * 200))
    for k in ("annualized_return", "annualized_volatility", "sharpe", "sortino",
              "max_drawdown", "calmar", "annualized_turnover", "n_days"):
        assert k in out


@pytest.fixture(scope="module")
def end_to_end_predictions():
    world = make_synthetic_world(2012, 2017)
    samples, _ = build_event_samples(
        world["filings"], world["prices"], world["calendar"], world["benchmarks"],
        train_end_date="2014-12-31", validation_end_date="2015-12-31",
    )
    scaled = compute_ex_ante_scale_for_samples(samples, world["prices"], world["calendar"],
                                               ScaleConfig())
    samples = apply_target(scaled, TargetConfig(k=2.5))
    enc = AnonymousFeatureEncoder(periods_back=4, min_company_count=2, min_sample_coverage_pct=0.1)
    enc.fit(samples[samples["split"] == "train"].dropna(subset=["target_normalized_signal"]),
            world["facts"])
    res = train_and_predict(
        "ridge", enc,
        samples[samples["split"] == "train"].dropna(subset=["target_normalized_signal"]),
        samples[samples["split"] == "validation"].dropna(subset=["target_normalized_signal"]),
        samples[samples["split"] == "test"].dropna(subset=["target_normalized_signal"]),
        world["facts"],
    )
    preds = attach_prediction_context(res.predictions, samples)
    preds = preds[preds["split"].isin(("validation", "test"))].dropna(subset=["entry_date", "exit_date"])
    return preds, world


def test_backtest_runs_end_to_end(end_to_end_predictions):
    preds, world = end_to_end_predictions
    cfg = BacktestConfig(start_date="2015-01-05", end_date="2018-06-29",
                         transaction_cost_bps_per_trade=10)
    pcfg = PortfolioConfig(min_positive_signal=0.0, max_single_stock_weight=0.5,
                           min_positions=1, target_positions=4, max_positions=4,
                           min_position_weight=0.0)
    res = run_backtest(preds, world["prices"], world["calendar"], cfg, pcfg,
                       RiskConfig(lookback_days=120))
    assert not res.daily.empty
    assert (res.daily["portfolio_value"] > 0).all()
    assert (res.daily["cash_weight"] >= -1e-9).all()
    assert (res.daily["cash_weight"] <= 1 + 1e-9).all()


def test_backtest_report_includes_benchmarks(end_to_end_predictions):
    preds, world = end_to_end_predictions
    cfg = BacktestConfig(start_date="2015-01-05", end_date="2018-06-29")
    pcfg = PortfolioConfig(min_positive_signal=0.0, max_single_stock_weight=0.5,
                           min_positions=1, target_positions=4, max_positions=4,
                           min_position_weight=0.0)
    res = run_backtest(preds, world["prices"], world["calendar"], cfg, pcfg,
                       RiskConfig(lookback_days=120))
    report = build_report(res, world["prices"], world["benchmarks"])
    assert "strategy" in report
    assert "benchmark_spy" in report
    assert "benchmark_equal_weight_universe" in report
    assert "benchmark_equal_risk_universe" in report
