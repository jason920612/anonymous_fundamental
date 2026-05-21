"""Phase 59: Michaelis-Menten signal saturation tests."""

from __future__ import annotations

import numpy as np

from afp.portfolio.mm_saturation import michaelis_menten


def test_zero_signal_returns_zero():
    out = michaelis_menten(np.array([0.0, 0.0, 0.0]))
    assert np.allclose(out, 0.0)


def test_small_signal_approximately_linear():
    # For signal << Km, output ~ signal/Km (linear)
    sig = np.array([0.001, 0.001, 0.001, 1.0, 1.0, 1.0, 0.0])
    out = michaelis_menten(sig)
    # The two smallest signals (0.001) get scaled the same way
    assert out[0] == out[1]


def test_saturation_at_high_signal():
    # explicit Km for clear semantics
    sig = np.array([0.01, 1000.0])
    out = michaelis_menten(sig, Km=1.0)
    # signal=1000, Km=1 → 1000/1001 ≈ 0.999
    assert out[1] > 0.99


def test_preserves_rank_order():
    sig = np.array([0.1, 0.3, 0.5, 0.7, 0.2])
    out = michaelis_menten(sig)
    assert np.all(np.argsort(out) == np.argsort(sig))


def test_negative_clipped_to_zero():
    out = michaelis_menten(np.array([-1.0, 0.5]))
    assert out[0] == 0.0
    assert out[1] > 0.0


def test_explicit_Km():
    sig = np.array([0.5])
    out = michaelis_menten(sig, Km=0.5, Vmax=1.0)
    # signal == Km → output = 0.5
    assert abs(out[0] - 0.5) < 1e-9
