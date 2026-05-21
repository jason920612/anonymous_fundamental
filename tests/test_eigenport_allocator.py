"""Phase 47: eigenportfolio market-mode subtraction tests."""

from __future__ import annotations

import numpy as np

from afp.portfolio.eigenport_allocator import project_out_eigenmode


def test_orthogonal_after_projection():
    """signal_orth · v1 should be ~zero."""
    v1 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    signal = np.array([0.3, 0.5, 0.7])
    s_orth = project_out_eigenmode(signal, v1)
    assert abs(s_orth @ v1) < 1e-9


def test_projection_preserves_non_market_information():
    """A signal purely orthogonal to v1 is unchanged."""
    v1 = np.array([1.0, 1.0]) / np.sqrt(2)
    signal = np.array([1.0, -1.0])
    s_orth = project_out_eigenmode(signal, v1)
    assert np.allclose(s_orth, signal)


def test_pure_market_signal_zeroes_out():
    """A signal parallel to v1 should reduce to zero after projection."""
    v1 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    signal = 2.0 * v1
    s_orth = project_out_eigenmode(signal, v1)
    assert np.allclose(s_orth, 0.0, atol=1e-9)
