"""Phase 14 — anonymous derived features (YoY, z-score)."""

import numpy as np
import pandas as pd
import pytest

from afp.features.derived import compute_derived
from afp.features.encoder import AnonymousFeatureEncoder, check_no_human_field_names
from afp.features.sample_builder import build_event_samples
from tests.fixtures import make_synthetic_world


def test_compute_derived_yoy_known_values():
    # Build a tiny [N=1, P=8, F=1] tensor with known values
    values = np.zeros((1, 8, 1))
    values[0, 0, 0] = 110.0
    values[0, 4, 0] = 100.0
    missing = np.zeros((1, 8, 1), dtype=np.int8)
    derived, derived_missing = compute_derived(values, missing, yoy_lookback=4)
    # YoY at index 0
    assert derived[0, 0, 0] == pytest.approx(10.0)
    assert derived_missing[0, 0, 0] == 0


def test_compute_derived_zscore_known_values():
    rng = np.random.default_rng(0)
    arr = rng.normal(50, 5, 8)
    values = arr.reshape(1, 8, 1)
    missing = np.zeros((1, 8, 1), dtype=np.int8)
    derived, _ = compute_derived(values, missing, yoy_lookback=4)
    # Second half (z block) = (v0 - mean) / std
    expected = (arr[0] - arr.mean()) / arr.std(ddof=1)
    assert derived[0, 0, 1] == pytest.approx(expected, rel=1e-9)


def test_compute_derived_handles_missing():
    values = np.array([[[100.0], [0.0], [0.0], [0.0], [100.0], [0.0], [0.0], [0.0]]])
    missing = np.array([[[0], [1], [1], [1], [0], [1], [1], [1]]], dtype=np.int8)
    derived, derived_missing = compute_derived(values, missing, yoy_lookback=4)
    assert derived_missing[0, 0, 0] == 0   # YoY available (both present)
    # z with only 2 valid observations is computable but high-variance; if std≈0 → masked
    # We just require it does not crash and the YoY is correct
    assert derived[0, 0, 0] == pytest.approx(0.0)


def test_encoder_with_derived_doubles_feature_count_in_columns():
    world = make_synthetic_world(2012, 2017)
    samples, _ = build_event_samples(world["filings"], world["prices"], world["calendar"],
                                     world["benchmarks"])
    train = samples[samples["split"] == "train"]
    enc = AnonymousFeatureEncoder(periods_back=4, min_company_count=2,
                                  min_sample_coverage_pct=0.1,
                                  derived_enabled=True, derived_yoy_lookback=2)
    enc.fit(train, world["facts"])
    scaled, missing, meta, sids = enc.transform(train, world["facts"])
    dense = enc.to_dense_frame(scaled, missing, meta, sids)
    base_n = len(enc.artifact.feature_map.feature_ids)
    derived_yoy_cols = [c for c in dense.columns if c.startswith("derived_yoy_")]
    derived_z_cols = [c for c in dense.columns if c.startswith("derived_z_")]
    # Each derived has _o0..._oP-1, multiplied by 2 (value + missing) = 2 * P each
    assert len(derived_yoy_cols) == 2 * base_n * enc.cfg.periods_back
    assert len(derived_z_cols) == 2 * base_n * enc.cfg.periods_back
    assert check_no_human_field_names(list(dense.columns)) == []


def test_encoder_save_load_preserves_derived_flag(tmp_path):
    world = make_synthetic_world(2012, 2017)
    samples, _ = build_event_samples(world["filings"], world["prices"], world["calendar"],
                                     world["benchmarks"])
    train = samples[samples["split"] == "train"]
    enc = AnonymousFeatureEncoder(periods_back=4, min_company_count=2,
                                  min_sample_coverage_pct=0.1,
                                  derived_enabled=True, derived_yoy_lookback=2)
    enc.fit(train, world["facts"])
    enc.save(tmp_path)
    enc2 = AnonymousFeatureEncoder.load(tmp_path)
    assert enc2.artifact.derived_enabled is True
    a, _, _, _ = enc.transform(train, world["facts"])
    b, _, _, _ = enc2.transform(train, world["facts"])
    assert np.allclose(a, b, equal_nan=True)
