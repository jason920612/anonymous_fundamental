"""Phase 3 tests — sample builder, date ordering, no-future-filing invariants."""

import math

import pandas as pd
import pytest

from afp.features.sample_builder import build_event_samples, required_sample_columns
from tests.fixtures import make_synthetic_world


@pytest.fixture(scope="module")
def world():
    return make_synthetic_world(2012, 2017)


def test_samples_have_required_columns(world):
    samples, _ = build_event_samples(world["filings"], world["prices"], world["calendar"],
                                     world["benchmarks"])
    for col in required_sample_columns():
        assert col in samples.columns


def test_entry_date_strictly_after_report_event(world):
    samples, _ = build_event_samples(world["filings"], world["prices"], world["calendar"],
                                     world["benchmarks"])
    assert (pd.to_datetime(samples["entry_date"]) > pd.to_datetime(samples["report_event_date"])).all()


def test_exit_date_before_next_report_event(world):
    samples, _ = build_event_samples(world["filings"], world["prices"], world["calendar"],
                                     world["benchmarks"])
    assert (pd.to_datetime(samples["exit_date"]) < pd.to_datetime(samples["next_report_event_date"])).all()


def test_entry_before_exit_strict(world):
    samples, _ = build_event_samples(world["filings"], world["prices"], world["calendar"],
                                     world["benchmarks"])
    assert (pd.to_datetime(samples["entry_date"]) < pd.to_datetime(samples["exit_date"])).all()


def test_raw_log_return_matches_prices(world):
    samples, _ = build_event_samples(world["filings"], world["prices"], world["calendar"],
                                     world["benchmarks"])
    s = samples.iloc[0]
    assert s["raw_log_return"] == pytest.approx(math.log(s["exit_price"] / s["entry_price"]))


def test_excluded_samples_record_reason(world):
    _, excluded = build_event_samples(world["filings"], world["prices"], world["calendar"],
                                      world["benchmarks"])
    # Each company contributes at least one missing_next_event row (the most recent filing)
    assert (excluded["exclusion_reason"] == "missing_next_event").sum() >= 1


def test_splits_assigned_by_entry_date(world):
    samples, _ = build_event_samples(world["filings"], world["prices"], world["calendar"],
                                     world["benchmarks"],
                                     train_end_date="2014-12-31",
                                     validation_end_date="2016-12-31")
    g = samples.groupby("split")["entry_date"].agg(["min", "max"])
    assert pd.Timestamp(g.loc["train", "max"]).date() <= pd.Timestamp("2014-12-31").date()
    assert pd.Timestamp(g.loc["validation", "min"]).date() > pd.Timestamp("2014-12-31").date()
