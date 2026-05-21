"""Phase 49: adaptive γ via signal entropy tests."""

from __future__ import annotations

import numpy as np

from afp.portfolio.adaptive_gamma import adaptive_gamma, signal_entropy


def test_uniform_signal_max_entropy_gives_gamma_max():
    sig = np.array([0.5, 0.5, 0.5, 0.5])
    g = adaptive_gamma(sig, gamma_min=2.0, gamma_max=10.0)
    assert abs(g - 10.0) < 0.01


def test_concentrated_signal_min_entropy_gives_gamma_min():
    sig = np.array([1.0, 0.0001, 0.0001, 0.0001])
    g = adaptive_gamma(sig, gamma_min=2.0, gamma_max=10.0)
    # Entropy is small but not exactly zero due to the small floor
    assert g < 4.0


def test_intermediate_dispersion_gives_intermediate_gamma():
    sig = np.array([0.4, 0.3, 0.2, 0.1])
    g = adaptive_gamma(sig, gamma_min=2.0, gamma_max=10.0)
    assert 6.0 < g < 10.0


def test_signal_entropy_zero_for_single_winner():
    sig = np.array([1.0, 0.0, 0.0])
    assert signal_entropy(sig) == 0.0


def test_signal_entropy_log_n_for_uniform():
    sig = np.ones(5)
    assert abs(signal_entropy(sig) - np.log(5)) < 1e-9


def test_negative_or_empty_input_returns_zero():
    assert signal_entropy(np.array([])) == 0.0
    assert signal_entropy(np.array([-1.0, -1.0])) == 0.0
