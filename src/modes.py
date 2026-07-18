"""Normal-mode decomposition of return covariance matrices.

Physics analogy
---------------
Consider ``N`` coupled harmonic oscillators with equations of motion

    ẍ_i = -Σ_j K_ij x_j

where ``K`` is a symmetric positive-definite coupling matrix. Diagonalizing
``K`` produces its eigenvectors — the **normal modes** — and eigenvalues
``ω_k²``. Each mode ``ξ_k = Σ_i v_ik x_i`` evolves independently:

    ξ̈_k = -ω_k² ξ_k

so the system's dynamics decompose into ``N`` independent oscillators.

In markets, the analogue is the return covariance ``Σ``. Its eigenvectors
``v_k`` are portfolios whose returns are (in-sample) uncorrelated:

    r_k(t) = Σ_i v_ik r_i(t),      Cov(r_k, r_l) = λ_k δ_kl

The eigenvalues play the role of variances rather than restoring
frequencies, but the diagonalization structure is identical.

Signal vs. noise: Marchenko–Pastur
----------------------------------
For a random covariance matrix constructed from ``T`` i.i.d. samples of
``N`` unit-variance variables, the eigenvalues in the ``T, N → ∞`` limit
lie in the bulk

    λ ∈ [σ²(1 - √q)², σ²(1 + √q)²],    q = N / T

where ``σ² = 1`` for standardized returns. Eigenvalues above the upper
bound carry statistically significant structure beyond noise. We call
these the **signal modes**.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class NormalModes:
    """Result of a normal-mode decomposition of a return covariance matrix.

    Attributes
    ----------
    eigenvalues : np.ndarray, shape (N,)
        Eigenvalues in descending order. Each is the variance carried by
        its corresponding mode (in units of the input covariance).
    eigenvectors : np.ndarray, shape (N, N)
        Eigenvectors as columns. Column ``k`` is the portfolio weight
        vector defining mode ``k``. Orthonormal: ``V.T @ V = I``.
    tickers : list[str]
        Asset identifiers, aligned with rows of ``eigenvectors``.
    mp_upper : float
        Marchenko–Pastur upper bound. Modes with eigenvalue at or below
        this are statistically indistinguishable from noise.
    n_signal_modes : int
        Number of modes strictly above ``mp_upper``.
    q : float
        Aspect ratio ``N / T`` used in the MP calculation.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    tickers: list[str] = field(default_factory=list)
    mp_upper: float = 0.0
    n_signal_modes: int = 0
    q: float = 0.0

    def signal_modes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (eigenvalues, eigenvectors) of modes above the MP bound."""
        mask = self.eigenvalues > self.mp_upper
        return self.eigenvalues[mask], self.eigenvectors[:, mask]

    def project(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Project a return series onto normal-mode coordinates.

        Each column of the returned DataFrame is the return series of the
        portfolio defined by the corresponding mode's eigenvector.

        Parameters
        ----------
        returns : pd.DataFrame
            Rows are dates, columns are assets. Must contain all tickers
            in ``self.tickers``; columns are reindexed to match.
        """
        R = returns[self.tickers].values  # (T, N)
        mode_returns = R @ self.eigenvectors  # (T, N)
        cols = [f"mode_{i}" for i in range(mode_returns.shape[1])]
        return pd.DataFrame(mode_returns, index=returns.index, columns=cols)

    def top_mode_weights(self, k: int = 0) -> pd.Series:
        """Return the weight vector of the k-th eigenmode as a labeled Series."""
        return pd.Series(self.eigenvectors[:, k], index=self.tickers, name=f"mode_{k}")


def marchenko_pastur_bounds(
    T: int, N: int, sigma2: float = 1.0
) -> tuple[float, float]:
    """Marchenko–Pastur lower and upper eigenvalue bounds.

    For a sample covariance matrix built from ``T`` i.i.d. samples of ``N``
    variables with variance ``σ²``, the bulk of the eigenvalue distribution
    is asymptotically supported on

        [σ²(1 - √q)², σ²(1 + √q)²]

    where ``q = N / T``. Requires ``q < 1`` for a non-degenerate bulk;
    when ``N > T`` the sample covariance is rank-deficient and the bulk
    formula must be modified.

    Parameters
    ----------
    T : int
        Number of samples (time observations).
    N : int
        Number of variables (assets).
    sigma2 : float
        Underlying variance. Use ``1.0`` for standardized returns.
    """
    if T <= 0 or N <= 0:
        raise ValueError(f"T and N must be positive, got T={T}, N={N}.")
    q = N / T
    lam_minus = sigma2 * (1 - np.sqrt(q)) ** 2
    lam_plus = sigma2 * (1 + np.sqrt(q)) ** 2
    return lam_minus, lam_plus


def decompose(returns: pd.DataFrame, standardize: bool = True) -> NormalModes:
    """Compute the normal-mode decomposition of a return covariance matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Rows are dates, columns are assets. Must be aligned and NaN-free.
    standardize : bool, default True
        If True, z-score each asset before computing the correlation
        matrix. This puts every asset on unit variance so that the
        Marchenko–Pastur bound applies directly with ``σ² = 1``. If
        False, use the raw covariance; the MP bound is then computed
        against the mean cross-sectional variance.

    Returns
    -------
    NormalModes
        Decomposition sorted in descending order of eigenvalue,
        annotated with the MP noise floor.
    """
    if returns.isnull().any().any():
        raise ValueError("Returns contain NaN; align and drop them first.")

    T, N = returns.shape
    if T < 2:
        raise ValueError(f"Need at least 2 rows to estimate covariance, got T={T}.")

    if standardize:
        R = (returns - returns.mean()) / returns.std(ddof=1)
        C = R.cov(ddof=1).values
        sigma2 = 1.0
    else:
        C = returns.cov(ddof=1).values
        sigma2 = float(returns.var(ddof=1).mean())

    # Symmetric eigendecomposition; eigh returns ascending order.
    eigvals, eigvecs = np.linalg.eigh(C)
    # Flip to descending.
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    # Ensure a deterministic sign convention: largest-magnitude entry positive.
    for k in range(eigvecs.shape[1]):
        i_max = int(np.argmax(np.abs(eigvecs[:, k])))
        if eigvecs[i_max, k] < 0:
            eigvecs[:, k] *= -1

    _, mp_upper = marchenko_pastur_bounds(T, N, sigma2)
    n_signal = int(np.sum(eigvals > mp_upper))

    return NormalModes(
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        tickers=list(returns.columns),
        mp_upper=float(mp_upper),
        n_signal_modes=n_signal,
        q=N / T,
    )


def clean_covariance(modes: NormalModes) -> np.ndarray:
    """Marchenko–Pastur cleaned covariance matrix.

    Eigenvalues above the MP upper bound are kept; eigenvalues at or below
    are replaced by their average so the trace is preserved. This is the
    standard RMT covariance shrinkage from Laloux, Cizeau, Bouchaud &
    Potters (1999).
    """
    lam = modes.eigenvalues.copy()
    V = modes.eigenvectors
    noise_mask = lam <= modes.mp_upper
    if noise_mask.any():
        lam[noise_mask] = lam[noise_mask].mean()
    return V @ np.diag(lam) @ V.T
