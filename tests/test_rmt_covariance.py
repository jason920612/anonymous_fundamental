"""Phase 38: Marchenko-Pastur covariance cleaning tests."""

from __future__ import annotations

import numpy as np

from afp.portfolio.rmt_covariance import (
    clean_correlation_rmt,
    clean_covariance_rmt,
    marchenko_pastur_lower,
    marchenko_pastur_upper,
    min_variance_weights,
    signal_tilted_min_var_weights,
)


def test_mp_bounds_match_textbook():
    # q=0.25 → (1±0.5)² = 2.25, 0.25
    assert marchenko_pastur_upper(0.25) == 2.25
    assert marchenko_pastur_lower(0.25) == 0.25
    # q=1 → upper 4, lower 0
    assert marchenko_pastur_upper(1.0) == 4.0
    assert marchenko_pastur_lower(1.0) == 0.0


def test_pure_noise_correlation_gets_squashed():
    """Pure iid Gaussian → no eigenvalues should pop above λ_+ (in expectation)."""
    rng = np.random.default_rng(0)
    N, T = 30, 300
    X = rng.standard_normal((T, N))
    C = np.corrcoef(X.T)
    C_clean = clean_correlation_rmt(C, T=T, keep_market_mode=True)
    # All cleaned eigenvalues (except the market mode) should be roughly equal:
    eigs = np.linalg.eigvalsh(C_clean)
    # The smallest N-1 should be nearly constant (the noise-mean replacement)
    assert eigs[:-1].std() < 0.1


def test_market_mode_preserved():
    """When all assets share a market factor, the top eigenvalue is informative."""
    rng = np.random.default_rng(1)
    N, T = 20, 400
    market = rng.standard_normal((T, 1))
    noise = rng.standard_normal((T, N)) * 0.3
    X = market + noise  # all assets loaded on the market
    cov = clean_covariance_rmt(X, keep_market_mode=True)
    eigs = np.linalg.eigvalsh(cov)
    # The dominant eigenvalue should be much larger than the rest.
    assert eigs[-1] > 5 * eigs[-2]


def test_min_variance_weights_sum_to_one_and_nonneg():
    rng = np.random.default_rng(2)
    N, T = 10, 250
    X = rng.standard_normal((T, N))
    cov = clean_covariance_rmt(X)
    w = min_variance_weights(cov, long_only=True, max_weight=0.5)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all()


def test_signal_tilted_min_var_respects_signal():
    """Higher-signal asset should get higher weight when σ is identical."""
    N = 5
    cov = np.eye(N) * 0.01
    signal = np.array([0.5, 0.2, 0.1, 0.0, 0.0])
    w = signal_tilted_min_var_weights(cov, signal, risk_aversion=2.0,
                                      long_only=True, max_weight=0.6)
    assert w[0] > w[1] > w[2]
    assert w.sum() > 0.99  # should be ~1
