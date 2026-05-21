"""Phase 42 candidate: HRP allocator tests."""

from __future__ import annotations

import numpy as np

from afp.portfolio.hrp_allocator import _corr_distance, hrp_weights


def test_corr_distance_is_proper_metric():
    rho = 0.4
    d = float(_corr_distance(np.array([[1.0, rho], [rho, 1.0]]))[0, 1])
    expected = np.sqrt(0.5 * (1 - rho))
    assert abs(d - expected) < 1e-9


def test_hrp_weights_sum_to_one():
    rng = np.random.default_rng(0)
    N, T = 8, 250
    X = rng.standard_normal((T, N))
    cov = np.cov(X.T)
    w = hrp_weights(cov)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w > 0).all()


def test_hrp_clusters_correlated_assets():
    """Two assets perfectly correlated with each other and uncorrelated with the rest
    should each receive ~half the weight that goes to that 'cluster'."""
    rng = np.random.default_rng(1)
    N, T = 6, 400
    common = rng.standard_normal((T, 1))
    paired = common + rng.standard_normal((T, 2)) * 0.05
    other = rng.standard_normal((T, 4))
    X = np.concatenate([paired, other], axis=1)
    cov = np.cov(X.T)
    w = hrp_weights(cov)
    # The two paired assets should have ~equal weight to each other.
    assert abs(w[0] - w[1]) < 0.05


def test_hrp_single_asset():
    cov = np.array([[0.04]])
    w = hrp_weights(cov)
    assert len(w) == 1
    assert w[0] == 1.0
