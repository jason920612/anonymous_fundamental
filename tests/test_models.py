"""Phase 6 tests — baselines train, predictions clipped, schema correct."""

import numpy as np
import pandas as pd
import pytest

from afp.features.encoder import AnonymousFeatureEncoder
from afp.features.sample_builder import build_event_samples
from afp.models.baselines import build_model
from afp.models.train import attach_prediction_context, train_and_predict
from afp.targets.target_transform import TargetConfig, apply as apply_target
from afp.targets.volatility_scale import ScaleConfig, compute_ex_ante_scale_for_samples
from tests.fixtures import make_synthetic_world


@pytest.fixture(scope="module")
def full_world():
    world = make_synthetic_world(2012, 2017)
    samples, _ = build_event_samples(
        world["filings"], world["prices"], world["calendar"], world["benchmarks"],
        train_end_date="2014-12-31", validation_end_date="2016-06-30",
    )
    scaled = compute_ex_ante_scale_for_samples(samples, world["prices"],
                                               world["calendar"], ScaleConfig())
    samples = apply_target(scaled, TargetConfig(k=2.5))
    encoder = AnonymousFeatureEncoder(periods_back=4, min_company_count=2,
                                      min_sample_coverage_pct=0.1)
    train = samples[samples["split"] == "train"].dropna(subset=["target_normalized_signal"])
    encoder.fit(train, world["facts"])
    return {
        "samples": samples,
        "facts": world["facts"],
        "encoder": encoder,
        "train": train,
        "val": samples[samples["split"] == "validation"].dropna(subset=["target_normalized_signal"]),
        "test": samples[samples["split"] == "test"].dropna(subset=["target_normalized_signal"]),
    }


@pytest.mark.parametrize("kind", ["constant_zero", "historical_mean", "ridge", "gbt"])
def test_baseline_trains_and_predicts_in_range(full_world, kind):
    res = train_and_predict(kind, full_world["encoder"], full_world["train"],
                            full_world["val"], full_world["test"], full_world["facts"])
    assert res.predictions["predicted_signal"].between(-1, 1).all()
    assert {"sample_id", "predicted_signal", "split", "model_id"} <= set(res.predictions.columns)


def test_constant_zero_baseline_is_exactly_zero(full_world):
    res = train_and_predict("constant_zero", full_world["encoder"], full_world["train"],
                            full_world["val"], full_world["test"], full_world["facts"])
    assert (res.predictions["predicted_signal"] == 0).all()


def test_validation_metrics_present(full_world):
    res = train_and_predict("ridge", full_world["encoder"], full_world["train"],
                            full_world["val"], full_world["test"], full_world["facts"])
    for k in ("mse", "mae", "direction_accuracy", "r2"):
        assert k in res.validation_metrics


def test_attach_prediction_context_adds_dates_and_scale(full_world):
    res = train_and_predict("ridge", full_world["encoder"], full_world["train"],
                            full_world["val"], full_world["test"], full_world["facts"])
    joined = attach_prediction_context(res.predictions, full_world["samples"])
    for col in ("entry_date", "exit_date", "ex_ante_scale"):
        assert col in joined.columns
