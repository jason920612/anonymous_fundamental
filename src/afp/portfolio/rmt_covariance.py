"""Random Matrix Theory covariance cleaning (Marchenko-Pastur).

Theory: for an N×N sample correlation matrix built from N×T iid Gaussian
returns, the eigenvalue density follows the Marchenko-Pastur distribution
supported on [(1-√q)², (1+√q)²] where q = N/T. Eigenvalues inside this
window are statistically indistinguishable from noise; eigenvalues above
λ_+ carry real structural information.

The standard cleaning recipe (Laloux, Cizeau, Bouchaud, Potters 1999):

1. Compute the correlation matrix C.
2. Eigen-decompose C = V Λ Vᵀ.
3. Keep the largest eigenvalue λ₁ (market mode) and all eigenvalues
   above λ_+. Replace every other eigenvalue with their mean, so that
   trace(C) is preserved.
4. Reconstruct C_clean = V Λ_clean Vᵀ and re-symmetrize / rescale to
   unit diagonal.
5. Cov_clean = D · C_clean · D where D = diag(σ_i).

The function is fully parameter-free — q is determined by the data
shape. There is nothing here to tune on val/test data.
"""

from __future__ import annotations

import numpy as np


def marchenko_pastur_upper(q: float, sigma2: float = 1.0) -> float:
    """Upper edge of the M-P spectrum for variance σ²."""
    return sigma2 * (1.0 + np.sqrt(q)) ** 2


def marchenko_pastur_lower(q: float, sigma2: float = 1.0) -> float:
    """Lower edge of the M-P spectrum for variance σ²."""
    return sigma2 * max(0.0, (1.0 - np.sqrt(q)) ** 2)


def clean_correlation_rmt(corr: np.ndarray, T: int, keep_market_mode: bool = True) -> np.ndarray:
    """Clean a correlation matrix with the Marchenko-Pastur eigenvalue filter.

    Parameters
    ----------
    corr : (N, N) symmetric correlation matrix.
    T    : number of return observations used to build `corr`.
    keep_market_mode :
        Always keep the largest eigenvalue even if it exceeds λ_+.
        This is the standard recipe — the top mode is the "market" and
        is treated as informative regardless of where it lands.

    Returns
    -------
    (N, N) cleaned correlation matrix with unit diagonal.
    """
    N = corr.shape[0]
    if N < 2 or T < 4:
        return corr.copy()

    corr = (corr + corr.T) / 2.0
    # eigh returns ascending eigenvalues
    eigvals, eigvecs = np.linalg.eigh(corr)
    q = N / T
    lam_plus = (1.0 + np.sqrt(q)) ** 2

    # Identify which eigenvalues to keep. The top one (last in ascending
    # order) is treated as the market mode and always retained.
    keep_mask = eigvals > lam_plus
    if keep_market_mode:
        keep_mask[-1] = True

    noise_mask = ~keep_mask
    if noise_mask.any():
        noise_avg = float(eigvals[noise_mask].mean())
        cleaned = eigvals.copy()
        cleaned[noise_mask] = noise_avg
    else:
        cleaned = eigvals.copy()

    C = eigvecs @ np.diag(cleaned) @ eigvecs.T
    C = (C + C.T) / 2.0
    # Rescale to unit diagonal so this remains a correlation matrix.
    d = np.sqrt(np.maximum(np.diag(C), 1e-12))
    C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C


def clean_covariance_rmt(returns: np.ndarray, keep_market_mode: bool = True,
                         ridge: float = 1e-6) -> np.ndarray:
    """Build RMT-cleaned covariance matrix from a (T, N) returns matrix.

    Parameters
    ----------
    returns : (T, N) daily returns. Rows are days, columns are assets.
    keep_market_mode : see `clean_correlation_rmt`.
    ridge : small diagonal added to the cleaned cov for invertibility.

    Returns
    -------
    (N, N) cleaned covariance matrix.
    """
    T, N = returns.shape
    if T < 4 or N < 2:
        return np.cov(returns.T) if N > 1 else np.array([[float(np.var(returns))]])
    # Per-asset std for converting between corr and cov.
    sigma = np.std(returns, axis=0, ddof=1)
    sigma = np.where(sigma < 1e-9, 1e-9, sigma)
    Z = (returns - returns.mean(axis=0)) / sigma  # standardized
    corr = (Z.T @ Z) / (T - 1)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    corr_clean = clean_correlation_rmt(corr, T=T, keep_market_mode=keep_market_mode)
    cov = corr_clean * np.outer(sigma, sigma)
    cov = (cov + cov.T) / 2.0
    cov += ridge * np.eye(N) * np.mean(sigma) ** 2
    return cov


def min_variance_weights(cov: np.ndarray, long_only: bool = True,
                         max_weight: float | None = None) -> np.ndarray:
    """Long-only min-variance weights via simple projection.

    Solves: minimize wᵀΣw s.t. Σw = 1, w ≥ 0.

    For long-only the closed form Σ⁻¹1 / (1ᵀΣ⁻¹1) is not guaranteed
    non-negative, so we iterate: drop the most-negative weight, solve
    again on the remaining assets, repeat until all weights ≥ 0.
    """
    N = cov.shape[0]
    if N == 0:
        return np.array([])
    active = np.arange(N)
    for _ in range(N):
        sub = cov[np.ix_(active, active)]
        try:
            inv = np.linalg.inv(sub + 1e-10 * np.eye(len(active)))
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(sub)
        ones = np.ones(len(active))
        denom = float(ones @ inv @ ones)
        if denom <= 0:
            break
        w_sub = inv @ ones / denom
        if long_only and (w_sub < -1e-9).any():
            # Drop the most-negative asset and retry.
            drop_local = int(np.argmin(w_sub))
            active = np.delete(active, drop_local)
            if len(active) == 0:
                break
            continue
        w = np.zeros(N)
        w[active] = np.clip(w_sub, 0.0, None)
        if w.sum() > 0:
            w = w / w.sum()
        if max_weight is not None:
            w = _cap_iter(w, max_weight)
        return w
    # Fallback: equal-weight on remaining active.
    w = np.zeros(N)
    if len(active) > 0:
        w[active] = 1.0 / len(active)
    return w


def signal_tilted_min_var_weights(cov: np.ndarray, signal: np.ndarray,
                                  risk_aversion: float = 5.0,
                                  long_only: bool = True,
                                  max_weight: float | None = None) -> np.ndarray:
    """Markowitz-style max(wᵀμ - γ/2 wᵀΣw) subject to w≥0, Σw=1.

    Uses a simple projected-gradient iteration. `risk_aversion` (γ) is
    set high enough that the result is close to min-variance but tilted
    by the model signal. γ=5 is a conventional choice in the literature
    when μ has unit scale; we do not tune it on val/test.
    """
    N = cov.shape[0]
    if N == 0:
        return np.array([])
    w = np.full(N, 1.0 / N)
    cov_psd = cov + 1e-10 * np.eye(N)
    # Lipschitz constant ≈ γ * λ_max(Σ).
    lam_max = float(np.linalg.eigvalsh(cov_psd).max())
    step = 1.0 / max(risk_aversion * lam_max, 1.0)
    for _ in range(500):
        grad = -signal + risk_aversion * (cov_psd @ w)
        w_new = w - step * grad
        if long_only:
            w_new = np.clip(w_new, 0.0, None)
        s = w_new.sum()
        if s > 0:
            w_new = w_new / s
        if np.max(np.abs(w_new - w)) < 1e-7:
            w = w_new
            break
        w = w_new
    if max_weight is not None:
        w = _cap_iter(w, max_weight)
    return w


def _cap_iter(w: np.ndarray, cap: float) -> np.ndarray:
    """Iteratively cap and renormalize (matches `allocation._cap_weights`)."""
    w = w.copy()
    n = len(w)
    if n == 0:
        return w
    for _ in range(20):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = w[over].sum() - cap * over.sum()
        w[over] = cap
        free = ~over
        if free.sum() == 0 or w[free].sum() == 0:
            break
        w[free] += excess * w[free] / w[free].sum()
    if w.sum() > 0:
        w = w / w.sum()
    return w
