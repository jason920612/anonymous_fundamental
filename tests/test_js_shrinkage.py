"""Phase 48: James-Stein shrinkage tests."""

from __future__ import annotations

import numpy as np

from afp.portfolio.js_shrinkage import js_shrink


def test_shrunken_signal_has_same_mean():
    sig = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    out = js_shrink(sig)
    assert abs(out.mean() - sig.mean()) < 1e-9


def test_shrunken_signal_has_lower_variance():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(20) * 0.5 + 0.5
    out = js_shrink(sig)
    assert out.var(ddof=1) <= sig.var(ddof=1)


def test_small_universe_no_shrinkage():
    # N < 3 → return identity
    sig = np.array([0.1, 0.5])
    out = js_shrink(sig)
    assert np.allclose(out, sig)


def test_constant_signal_unchanged():
    sig = np.array([0.5, 0.5, 0.5, 0.5])
    out = js_shrink(sig)
    assert np.allclose(out, sig)


def test_shrinkage_preserves_rank():
    sig = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    out = js_shrink(sig)
    assert np.all(np.argsort(out) == np.argsort(sig))
