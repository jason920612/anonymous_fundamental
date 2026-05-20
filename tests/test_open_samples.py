"""Phase 35.1 — open-sample builder for forward prediction."""

import pandas as pd
import pytest

from afp.features.sample_builder import build_event_samples, build_open_samples
from tests.fixtures import make_synthetic_world


@pytest.fixture(scope="module")
def world():
    return make_synthetic_world(2012, 2017)


def test_open_samples_one_per_company(world):
    open_samples = build_open_samples(world["filings"], world["prices"], world["calendar"])
    assert not open_samples.empty
    # one per company
    assert open_samples["internal_company_id"].nunique() == len(open_samples)


def test_open_samples_use_latest_filing(world):
    open_samples = build_open_samples(world["filings"], world["prices"], world["calendar"])
    for _, row in open_samples.iterrows():
        company_filings = world["filings"][world["filings"]["internal_company_id"] == row["internal_company_id"]]
        latest = company_filings.sort_values("accepted_datetime").iloc[-1]
        assert row["filing_event_id"] == latest["filing_event_id"]


def test_open_samples_have_no_realized_target(world):
    open_samples = build_open_samples(world["filings"], world["prices"], world["calendar"])
    assert open_samples["raw_log_return"].isna().all()
    assert open_samples["target_normalized_signal"].isna().all()
    assert (open_samples["split"] == "open").all()


def test_open_samples_disjoint_from_event_samples(world):
    closed, _ = build_event_samples(world["filings"], world["prices"], world["calendar"], world["benchmarks"])
    open_samples = build_open_samples(world["filings"], world["prices"], world["calendar"])
    # Latest filings of each company should NOT appear in closed (build_event_samples skips last)
    closed_latest = closed.sort_values("entry_date").groupby("internal_company_id").tail(1)
    for _, row in open_samples.iterrows():
        assert row["filing_event_id"] not in closed_latest["filing_event_id"].tolist() or \
               row["entry_date"] != closed_latest.loc[closed_latest["internal_company_id"] == row["internal_company_id"],
                                                       "entry_date"].iloc[0]
