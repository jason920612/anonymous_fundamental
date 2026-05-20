"""Phase 22: conditional DDPM for normalized target signal.

RFC-05 §7. The model learns the conditional distribution
`p(standardized_movement | anonymous_features)` and generates `N` samples
per event. The mean (or other distributional statistic) is then squashed
through `tanh / k` to recover a `predicted_signal ∈ [-1, 1]`.

Architecture
------------
- Condition encoder: small MLP (Linear→GELU→Dropout→Linear→GELU) producing
  a 128-dim latent vector per event.
- Denoiser ε-network: MLP that takes `[noisy_y, cond_latent, t_embed]` and
  predicts the noise.
- Schedule: linear β from 1e-4 to 0.02 over `T=200` steps.
- Loss: MSE on predicted noise.
- Inference: DDPM ancestral sampler, N samples per event, averaged.

All on PyTorch / CUDA when available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DiffusionConfig:
    timesteps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    cond_hidden: int = 256
    cond_latent: int = 128
    denoise_hidden: int = 256
    time_embed_dim: int = 64
    dropout: float = 0.2
    batch_size: int = 256
    epochs: int = 60
    lr: float = 5e-4
    weight_decay: float = 1e-4
    seed: int = 42
    standardize_target: bool = True
    inference_samples: int = 64
    clip_signal: float = 0.99


def _sinusoidal_time_embedding(t, dim):
    """Sinusoidal embedding ala Transformer / DDPM."""
    import torch
    device = t.device
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(half, device=device) / max(half - 1, 1))
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class _CondEncoder:
    def __init__(self):
        raise NotImplementedError


def _build_modules(n_features: int, cfg: DiffusionConfig):
    """Return `(condition_encoder, denoiser)` nn.Module instances."""
    import torch
    from torch import nn

    cond_enc = nn.Sequential(
        nn.Linear(n_features, cfg.cond_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
        nn.Linear(cfg.cond_hidden, cfg.cond_latent), nn.GELU(),
    )

    denoise_input_dim = 1 + cfg.cond_latent + cfg.time_embed_dim
    denoiser = nn.Sequential(
        nn.Linear(denoise_input_dim, cfg.denoise_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
        nn.Linear(cfg.denoise_hidden, cfg.denoise_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
        nn.Linear(cfg.denoise_hidden, 1),
    )
    return cond_enc, denoiser


class ConditionalDiffusion:
    """DDPM with continuous scalar target and high-dim conditioning."""

    def __init__(self, config: DiffusionConfig | None = None):
        self.cfg = config or DiffusionConfig()
        self.device = None
        self.cond_enc = None
        self.denoiser = None
        self.x_mean = 0.0
        self.x_std = 1.0
        self.y_mean = 0.0
        self.y_std = 1.0
        self.betas = None
        self.alphas = None
        self.alpha_bars = None

    def _build_schedule(self):
        import torch
        betas = torch.linspace(self.cfg.beta_start, self.cfg.beta_end, self.cfg.timesteps,
                                device=self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        import torch
        from torch import nn
        from torch.optim import AdamW

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_features = X.shape[1]

        X = np.nan_to_num(X.astype(np.float32))
        y = np.nan_to_num(y.astype(np.float32))
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-6
        Xs = (X - self.x_mean) / self.x_std

        if self.cfg.standardize_target:
            self.y_mean = float(y.mean())
            self.y_std = float(y.std()) + 1e-6
            y_std = (y - self.y_mean) / self.y_std
        else:
            y_std = y

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        self.cond_enc, self.denoiser = _build_modules(n_features, self.cfg)
        self.cond_enc.to(self.device)
        self.denoiser.to(self.device)
        self._build_schedule()

        X_t = torch.from_numpy(Xs).to(self.device)
        y_t = torch.from_numpy(y_std.astype(np.float32)).to(self.device).unsqueeze(-1)
        w_t = (torch.from_numpy(sample_weight.astype(np.float32)).to(self.device).unsqueeze(-1)
               if sample_weight is not None else None)

        opt = AdamW(list(self.cond_enc.parameters()) + list(self.denoiser.parameters()),
                    lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        loss_fn = nn.MSELoss(reduction="none")
        n = len(X_t)
        idx = np.arange(n)
        rng = np.random.default_rng(self.cfg.seed)

        for _ in range(self.cfg.epochs):
            rng.shuffle(idx)
            self.cond_enc.train(); self.denoiser.train()
            for start in range(0, n, self.cfg.batch_size):
                batch = idx[start:start + self.cfg.batch_size]
                xb = X_t[batch]; yb = y_t[batch]
                # Sample t uniform from [0, T-1]
                t = torch.randint(0, self.cfg.timesteps, (len(batch),), device=self.device)
                noise = torch.randn_like(yb)
                ab = self.alpha_bars[t].unsqueeze(-1)
                # q(y_t | y_0) = sqrt(ab) y_0 + sqrt(1-ab) noise
                yt = torch.sqrt(ab) * yb + torch.sqrt(1 - ab) * noise

                cond = self.cond_enc(xb)
                t_embed = _sinusoidal_time_embedding(t, self.cfg.time_embed_dim)
                inp = torch.cat([yt, cond, t_embed], dim=-1)
                pred_noise = self.denoiser(inp)
                l = loss_fn(pred_noise, noise)
                if w_t is not None:
                    l = (l * w_t[batch]).mean()
                else:
                    l = l.mean()
                opt.zero_grad()
                l.backward()
                opt.step()
        return self

    @property
    def model_type(self) -> str:
        return "diffusion"

    @staticmethod
    def _clip(p):
        return np.clip(p, -1.0, 1.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_distribution(X)["mean"]

    def predict_distribution(self, X: np.ndarray) -> dict:
        """Return per-sample mean / std / probability_positive / quantiles."""
        import torch
        X = np.nan_to_num(X.astype(np.float32))
        Xs = (X - self.x_mean) / self.x_std
        X_t = torch.from_numpy(Xs).to(self.device)
        n = len(X_t)
        S = self.cfg.inference_samples

        self.cond_enc.eval(); self.denoiser.eval()
        with torch.no_grad():
            cond_all = self.cond_enc(X_t)
            samples_y = []
            for _ in range(S):
                y = torch.randn(n, 1, device=self.device)
                for step in reversed(range(self.cfg.timesteps)):
                    t = torch.full((n,), step, device=self.device, dtype=torch.long)
                    t_embed = _sinusoidal_time_embedding(t, self.cfg.time_embed_dim)
                    inp = torch.cat([y, cond_all, t_embed], dim=-1)
                    eps = self.denoiser(inp)
                    a = self.alphas[step]; ab = self.alpha_bars[step]
                    coef = (1 - a) / torch.sqrt(1 - ab)
                    mean = (y - coef * eps) / torch.sqrt(a)
                    if step > 0:
                        noise = torch.randn_like(y)
                        sigma = torch.sqrt(self.betas[step])
                        y = mean + sigma * noise
                    else:
                        y = mean
                samples_y.append(y.squeeze(-1).cpu().numpy())
        samples_y = np.stack(samples_y, axis=0)  # [S, N]
        if self.cfg.standardize_target:
            samples_y = samples_y * self.y_std + self.y_mean

        mean = samples_y.mean(axis=0)
        std = samples_y.std(axis=0)
        prob_pos = (samples_y > 0).mean(axis=0)
        q05 = np.quantile(samples_y, 0.05, axis=0)
        q95 = np.quantile(samples_y, 0.95, axis=0)
        return {
            "mean": self._clip(mean),
            "std": std,
            "probability_positive": prob_pos,
            "q05": q05,
            "q95": q95,
        }
