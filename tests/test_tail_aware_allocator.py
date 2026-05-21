"""Phase 43 candidate: tail-aware (ES) allocator tests."""

from __future__ import annotations

import numpy as np

from afp.portfolio.tail_aware_allocator import (
    expected_shortfall,
    tail_risk_weights,
)


def test_expected_shortfall_positive_for_negative_tail():
    rng = np.random.default_rng(0)
    rets = rng.standard_normal(2000) * 0.01
    es = expected_shortfall(rets, alpha=0.05)
    # ES should be positive and roughly within reasonable bounds
    assert es > 0
    assert 0.01 < es < 0.05


def test_es_higher_for_heavy_tail():
    rng = np.random.default_rng(1)
    gauss = rng.standard_normal(2000) * 0.02
    student = rng.standard_t(df=3, size=2000) * 0.02
    es_g = expected_shortfall(gauss, alpha=0.05)
    es_t = expected_shortfall(student, alpha=0.05)
    assert es_t > es_g


def test_tail_risk_weights_simplex():
    es = np.array([0.02, 0.01, 0.03, 0.025])
    sig = np.array([0.5, 0.3, 0.7, 0.4])
    w = tail_risk_weights(es, sig)
    assert abs(w.sum() - 1.0) < 1e-9
    # Asset with smallest ES (highest "quality") and middling signal should
    # not be dominated by the highest-signal-but-highest-ES asset:
    # signal/ES:  0.5/0.02=25, 0.3/0.01=30, 0.7/0.03=23.3, 0.4/0.025=16
    # so asset 1 (smallest ES) should have the highest weight.
    assert np.argmax(w) == 1


def test_tail_risk_weights_uniform_if_no_signal():
    es = np.array([0.02, 0.02, 0.02])
    sig = np.array([0.0, 0.0, 0.0])
    w = tail_risk_weights(es, sig)
    assert np.allclose(w, [1/3, 1/3, 1/3])
