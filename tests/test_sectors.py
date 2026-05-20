"""Phase 16 — SIC sector parsing + sector caps."""

import numpy as np
import pandas as pd
import pytest

from afp.data.sectors import parse_sector_from_submissions, sic_to_division
from afp.portfolio.allocation import (
    PortfolioConfig,
    inverse_vol_allocation,
    select_candidates,
)


def test_sic_to_division_normalizes():
    assert sic_to_division(7372) == "7"     # services
    assert sic_to_division("3674") == "3"   # manufacturing
    assert sic_to_division(None) is None
    assert sic_to_division("") is None


def test_parse_sector_from_submissions():
    payload = {"cik": 320193, "sic": 3571, "sicDescription": "Electronic Computers"}
    out = parse_sector_from_submissions(payload)
    assert out["cik"] == "0000320193"
    assert out["internal_company_id"] == "COMP0000320193"
    assert out["sic"] == "3571"
    assert out["sic_division"] == "3"


def test_sector_cap_redistributes_excess_to_other_sectors():
    # 4 candidates all in sector "3" with one in sector "5"
    preds = pd.DataFrame([
        {"internal_company_id": f"A{i}", "predicted_signal": 0.5, "vol_estimate": 0.02, "sector": "3"}
        for i in range(4)
    ] + [
        {"internal_company_id": "B0", "predicted_signal": 0.5, "vol_estimate": 0.02, "sector": "5"},
    ])
    cfg = PortfolioConfig(min_positive_signal=0.0, max_single_stock_weight=0.5,
                          min_position_weight=0.0, allow_cash=False,
                          max_sector_weight=0.5)
    # select_candidates preserves sector column from preds
    cands = select_candidates(preds, cfg)
    assert "sector" in cands.columns
    res = inverse_vol_allocation(cands, cfg)
    s3 = res.weights[res.weights["internal_company_id"].isin(["A0","A1","A2","A3"])]["weight"].sum()
    s5 = res.weights[res.weights["internal_company_id"] == "B0"]["weight"].sum()
    assert s3 <= 0.5 + 1e-9
    assert s5 > 0.5 - 1e-9


def test_sector_cap_no_op_when_disabled():
    preds = pd.DataFrame([
        {"internal_company_id": f"A{i}", "predicted_signal": 0.5, "vol_estimate": 0.02, "sector": "3"}
        for i in range(5)
    ])
    cfg = PortfolioConfig(min_positive_signal=0.0, max_single_stock_weight=0.5,
                          min_position_weight=0.0, allow_cash=False,
                          max_sector_weight=None)
    cands = select_candidates(preds, cfg)
    res = inverse_vol_allocation(cands, cfg)
    assert res.weights["weight"].sum() > 0.99
