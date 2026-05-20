"""Phase 22 — conditional diffusion sanity tests (GPU optional)."""

import numpy as np
import pytest

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from afp.models.diffusion import ConditionalDiffusion, DiffusionConfig


@pytest.mark.skipif(not HAVE_TORCH, reason="torch unavailable")
def test_diffusion_trains_and_predicts_in_range():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 10)).astype(np.float32)
    # Target: tanh of first feature + noise
    y = np.tanh(X[:, 0] + rng.normal(0, 0.3, 300)).astype(np.float32)
    model = ConditionalDiffusion(DiffusionConfig(
        epochs=4, batch_size=64, timesteps=50,
        cond_hidden=64, cond_latent=32, denoise_hidden=64, time_embed_dim=16,
        inference_samples=8,
    ))
    model.fit(X, y)
    out = model.predict_distribution(X)
    assert out["mean"].shape == (300,)
    assert ((out["mean"] >= -1) & (out["mean"] <= 1)).all()
    assert out["probability_positive"].shape == (300,)
    assert (out["std"] >= 0).all()


@pytest.mark.skipif(not HAVE_TORCH, reason="torch unavailable")
def test_diffusion_distribution_has_real_spread():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)).astype(np.float32)
    y = (X[:, 0] * 0.3).astype(np.float32)
    model = ConditionalDiffusion(DiffusionConfig(
        epochs=2, batch_size=64, timesteps=30,
        cond_hidden=32, cond_latent=16, denoise_hidden=32, time_embed_dim=8,
        inference_samples=16,
    ))
    model.fit(X, y)
    out = model.predict_distribution(X)
    # Spread comes from the stochastic sampler; should be non-trivial.
    assert out["std"].mean() > 0.01
