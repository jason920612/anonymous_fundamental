"""Phase 30: tabular transformer over the period axis.

The encoder produces `[N, P=8, F]` values + missing + 3 metadata channels. We
reshape so each period becomes a token of dim `2F + 3` (value + missing + meta),
project to a model dim, add a learned per-period positional embedding, run a
small encoder transformer, mean-pool, and predict a tanh signal.

Anonymity preserved: tokens still carry only opaque numeric channels — no
human concept names enter the architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PeriodTransformerConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 3
    dropout: float = 0.2
    batch_size: int = 256
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42


class PeriodTransformer:
    """[N,P,F] → tanh signal. GPU when available."""

    model_type = "period_transformer"

    def __init__(self, config: Optional[PeriodTransformerConfig] = None):
        self.cfg = config or PeriodTransformerConfig()
        self.model = None
        self.device = None
        self.feature_count = None
        self.x_mean = 0.0
        self.x_std = 1.0
        self.y_mean = 0.0
        self.y_std = 1.0

    @staticmethod
    def _clip(p):
        return np.clip(p, -1.0, 1.0)

    def _build(self, token_dim: int):
        import torch
        from torch import nn

        class Block(nn.Module):
            def __init__(self, d_model, n_heads, dropout):
                super().__init__()
                self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
                self.ln1 = nn.LayerNorm(d_model)
                self.ff = nn.Sequential(
                    nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(4 * d_model, d_model),
                )
                self.ln2 = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                a, _ = self.attn(x, x, x, need_weights=False)
                x = self.ln1(x + self.dropout(a))
                x = self.ln2(x + self.dropout(self.ff(x)))
                return x

        d = self.cfg.d_model
        n_periods = 8
        proj = nn.Linear(token_dim, d)
        pos = nn.Parameter(torch.zeros(1, n_periods, d))
        blocks = nn.ModuleList([Block(d, self.cfg.n_heads, self.cfg.dropout)
                                for _ in range(self.cfg.n_layers)])
        head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(),
                             nn.Dropout(self.cfg.dropout), nn.Linear(d // 2, 1), nn.Tanh())

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = proj
                self.pos = pos
                self.blocks = blocks
                self.head = head
                self.dropout = nn.Dropout(self.cfg.dropout) if hasattr(self, 'cfg') else nn.Dropout(0.0)

        net = Net.__new__(Net)
        nn.Module.__init__(net)
        net.proj = proj
        net.pos = pos
        net.blocks = blocks
        net.head = head
        net.drop = nn.Dropout(self.cfg.dropout)

        def forward(self, tokens):
            x = self.proj(tokens) + self.pos
            x = self.drop(x)
            for b in self.blocks:
                x = b(x)
            pooled = x.mean(dim=1)
            return self.head(pooled)

        net.forward = forward.__get__(net, type(net))
        return net

    def _reshape_tokens(self, X_flat: np.ndarray) -> np.ndarray:
        """Take the build_tabular_dataset output and reshape to `[N, P=8, token_dim]`.

        The matrix layout (see `datasets.build_tabular_dataset`):
          [ values(P*F) | missing(P*F) | metadata(P*3) ]
        with values ordered as [(f0,o0..oP-1), (f1,o0..oP-1), ...].
        """
        N = X_flat.shape[0]
        P = 8
        F = self.feature_count
        meta = 3
        expected = P * F + P * F + P * meta
        if X_flat.shape[1] != expected:
            raise ValueError(f"unexpected X width: {X_flat.shape[1]} (expected {expected}); "
                             f"period transformer cannot reshape this layout")
        values = X_flat[:, : P * F].reshape(N, F, P).transpose(0, 2, 1)  # [N, P, F]
        missing = X_flat[:, P * F : 2 * P * F].reshape(N, F, P).transpose(0, 2, 1)
        meta_block = X_flat[:, 2 * P * F :].reshape(N, meta, P).transpose(0, 2, 1)  # [N, P, 3]
        tokens = np.concatenate([values, missing, meta_block], axis=2)  # [N, P, 2F+3]
        return tokens.astype(np.float32)

    def fit(self, X, y, sample_weight=None, feature_count: int | None = None):
        import torch
        from torch import nn
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        if feature_count is None:
            raise ValueError("PeriodTransformer.fit requires feature_count (base anonymous F)")
        self.feature_count = feature_count

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokens = self._reshape_tokens(X)
        token_dim = tokens.shape[-1]

        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) + 1e-6
        y_std = (y - self.y_mean) / self.y_std

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.model = self._build(token_dim).to(self.device)
        opt = AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sched = CosineAnnealingLR(opt, T_max=self.cfg.epochs)
        loss_fn = nn.SmoothL1Loss(reduction="none", beta=0.1)

        X_t = torch.from_numpy(tokens).to(self.device)
        y_t = torch.from_numpy(y_std.astype(np.float32)).to(self.device).unsqueeze(-1)
        w_t = (torch.from_numpy(sample_weight.astype(np.float32)).to(self.device).unsqueeze(-1)
               if sample_weight is not None else None)
        n = len(X_t)
        idx = np.arange(n)
        rng = np.random.default_rng(self.cfg.seed)
        for _ in range(self.cfg.epochs):
            rng.shuffle(idx)
            self.model.train()
            for start in range(0, n, self.cfg.batch_size):
                batch = idx[start:start + self.cfg.batch_size]
                xb = X_t[batch]; yb = y_t[batch]
                pred = self.model(xb)
                l = loss_fn(pred, yb)
                if w_t is not None:
                    l = (l * w_t[batch]).mean()
                else:
                    l = l.mean()
                opt.zero_grad()
                l.backward()
                opt.step()
            sched.step()
        return self

    def predict(self, X):
        import torch
        tokens = self._reshape_tokens(X)
        self.model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(tokens).to(self.device)
            outs = []
            for i in range(0, len(X_t), 4096):
                p = self.model(X_t[i:i + 4096]).cpu().numpy().ravel()
                outs.append(p * self.y_std + self.y_mean)
        return self._clip(np.concatenate(outs))
