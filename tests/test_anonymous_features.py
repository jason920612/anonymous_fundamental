"""Phase 5 tests — anonymity, stable IDs, train-only scaling, no future facts."""

import numpy as np
import pandas as pd
import pytest

from afp.features.encoder import AnonymousFeatureEncoder, check_no_human_field_names
from afp.features.sample_builder import build_event_samples
from tests.fixtures import make_synthetic_world


@pytest.fixture(scope="module")
def world():
    return make_synthetic_world(2012, 2017)


@pytest.fixture(scope="module")
def samples_and_facts(world):
    samples, _ = build_event_samples(world["filings"], world["prices"], world["calendar"],
                                     world["benchmarks"])
    return samples, world["facts"]


def test_encoder_fits_on_train_only(samples_and_facts):
    samples, facts = samples_and_facts
    train = samples[samples["split"] == "train"]
    enc = AnonymousFeatureEncoder(periods_back=4, min_company_count=2, min_sample_coverage_pct=0.1)
    enc.fit(train, facts)
    assert enc.artifact is not None
    assert enc.artifact.num_features() > 0


def test_dense_columns_have_no_human_field_names(samples_and_facts):
    samples, facts = samples_and_facts
    train = samples[samples["split"] == "train"]
    enc = AnonymousFeatureEncoder(periods_back=4, min_company_count=2, min_sample_coverage_pct=0.1)
    enc.fit(train, facts)
    scaled, missing, meta, sids = enc.transform(train, facts)
    dense = enc.to_dense_frame(scaled, missing, meta, sids)
    offenders = check_no_human_field_names(list(dense.columns))
    assert offenders == [], f"forbidden tokens leaked into model columns: {offenders}"


def test_feature_ids_are_stable_and_unique(samples_and_facts):
    samples, facts = samples_and_facts
    train = samples[samples["split"] == "train"]
    enc = AnonymousFeatureEncoder(periods_back=4, min_company_count=2, min_sample_coverage_pct=0.1)
    enc.fit(train, facts)
    fmap = enc.artifact.feature_map.table
    assert fmap["anonymous_feature_id"].is_unique
    # Each (taxonomy, concept_name, unit) → exactly one feature ID
    triples = fmap.groupby(["taxonomy", "concept_name", "unit"])["anonymous_feature_id"].nunique()
    assert (triples == 1).all()


def test_scaler_uses_only_train_statistics(samples_and_facts):
    """Mutating non-train facts must not change train-fitted scaling stats."""
    samples, facts = samples_and_facts
    train = samples[samples["split"] == "train"]
    non_train_ids = samples.loc[samples["split"] != "train", "sample_id"]
    enc1 = AnonymousFeatureEncoder(periods_back=4, min_company_count=2, min_sample_coverage_pct=0.1)
    enc1.fit(train, facts)
    facts_perturbed = facts.copy()
    # Perturb facts whose accepted_datetime is after train cutoff so train rows are unaffected.
    cutoff = train["accepted_datetime"].max()
    mask = facts_perturbed["accepted_datetime"] > cutoff
    facts_perturbed.loc[mask, "value"] = facts_perturbed.loc[mask, "value"].astype(float) * 1000.0
    enc2 = AnonymousFeatureEncoder(periods_back=4, min_company_count=2, min_sample_coverage_pct=0.1)
    enc2.fit(train, facts_perturbed)
    assert np.allclose(enc1.artifact.scaler.median, enc2.artifact.scaler.median, equal_nan=True)
    assert np.allclose(enc1.artifact.scaler.iqr, enc2.artifact.scaler.iqr, equal_nan=True)


def test_transform_does_not_use_future_facts(samples_and_facts):
    """Polluting facts accepted after each sample must not change its features."""
    samples, facts = samples_and_facts
    train = samples[samples["split"] == "train"]
    enc = AnonymousFeatureEncoder(periods_back=4, min_company_count=2, min_sample_coverage_pct=0.1)
    enc.fit(train, facts)
    scaled_a, _, _, _ = enc.transform(train, facts)

    facts_perturbed = facts.copy()
    latest_accepted = train["accepted_datetime"].max()
    mask = facts_perturbed["accepted_datetime"] > latest_accepted
    facts_perturbed.loc[mask, "value"] = facts_perturbed.loc[mask, "value"].astype(float) * 10.0
    scaled_b, _, _, _ = enc.transform(train, facts_perturbed)
    assert np.allclose(scaled_a, scaled_b, equal_nan=True)


def test_save_and_load_roundtrip(tmp_path, samples_and_facts):
    samples, facts = samples_and_facts
    train = samples[samples["split"] == "train"]
    enc = AnonymousFeatureEncoder(periods_back=4, min_company_count=2, min_sample_coverage_pct=0.1)
    enc.fit(train, facts)
    enc.save(tmp_path)
    enc2 = AnonymousFeatureEncoder.load(tmp_path)
    a, _, _, _ = enc.transform(train, facts)
    b, _, _, _ = enc2.transform(train, facts)
    assert np.allclose(a, b, equal_nan=True)
