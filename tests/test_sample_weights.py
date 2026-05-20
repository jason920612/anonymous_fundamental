"""Phase 15 — inverse-frequency sample weighting."""

from datetime import date

import numpy as np
import pandas as pd

from afp.models.baselines import build_model
from afp.models.train import compute_sample_weights


def _samples(rows):
    return pd.DataFrame(rows)


def test_weights_invert_company_frequency():
    samples = _samples([
        {"sample_id": f"A_{i}", "internal_company_id": "AAA", "entry_date": date(2020, 1, 1)}
        for i in range(8)
    ] + [
        {"sample_id": "B_0", "internal_company_id": "BBB", "entry_date": date(2020, 1, 1)},
        {"sample_id": "B_1", "internal_company_id": "BBB", "entry_date": date(2020, 1, 1)},
    ])
    w = compute_sample_weights(samples, samples["sample_id"].to_numpy(),
                                by_company=True, by_quarter=False)
    # AAA samples (8 rows) should get smaller weight than BBB samples (2 rows)
    w_aaa = w[samples["internal_company_id"] == "AAA"].mean()
    w_bbb = w[samples["internal_company_id"] == "BBB"].mean()
    assert w_bbb > w_aaa
    # Mean weight ≈ 1
    assert np.isclose(w.mean(), 1.0, atol=1e-9)


def test_weights_invert_quarter_frequency():
    samples = _samples([
        {"sample_id": f"S_q1_{i}", "internal_company_id": f"C{i}", "entry_date": date(2020, 1, 1)}
        for i in range(10)
    ] + [
        {"sample_id": "S_q2_0", "internal_company_id": "X", "entry_date": date(2020, 4, 1)},
    ])
    w = compute_sample_weights(samples, samples["sample_id"].to_numpy(),
                                by_company=False, by_quarter=True)
    q1_w = w[pd.to_datetime(samples["entry_date"]).dt.quarter == 1].mean()
    q2_w = w[pd.to_datetime(samples["entry_date"]).dt.quarter == 2].mean()
    assert q2_w > q1_w


def test_weights_none_when_flags_off():
    samples = _samples([
        {"sample_id": "S0", "internal_company_id": "A", "entry_date": date(2020, 1, 1)},
    ])
    w = compute_sample_weights(samples, samples["sample_id"].to_numpy(),
                                by_company=False, by_quarter=False)
    assert w is None


def test_ridge_accepts_sample_weight():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    y = X[:, 0] + rng.normal(0, 0.1, 50)
    m = build_model("ridge")
    m.fit(X, y, sample_weight=np.ones(50))
    assert m.predict(X).shape == (50,)


def test_gbt_accepts_sample_weight():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = X[:, 0] + rng.normal(0, 0.1, 100)
    m = build_model("gbt")
    m.fit(X, y, sample_weight=np.ones(100))
    assert m.predict(X).shape == (100,)
