"""Market regime detection with a Gaussian hidden Markov model.

Market-level features (computed from data through day ``d`` only):

* ``log_vol``    — log of 21-day realized volatility of the equal-weight
  market return (annualized).
* ``avg_corr``   — 63-day average pairwise correlation, estimated from
  the variance of the cross-sectional mean of volatility-standardized
  returns: for ``N`` unit-variance assets with average correlation
  ``ρ̄``, ``Var(mean_i z_i) = ρ̄ + (1 - ρ̄)/N``.
* ``log_disp``   — log of the 21-day average cross-sectional dispersion
  of returns.

A 2-state diagonal-covariance Gaussian HMM is fit by Baum–Welch on an
expanding window and refit once a year. The probabilities passed to
the strategy are **filtered** (forward-algorithm) probabilities, so
the regime probability on day ``d`` uses observations through ``d``
and parameters estimated on data before the last refit date — no
smoothing, no look-ahead. States are ordered by mean volatility, so
state 1 is always the high-volatility ("stress") state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def market_features(returns: pd.DataFrame) -> pd.DataFrame:
    """Daily market-level regime features (backward-looking)."""
    N = returns.shape[1]
    mkt = returns.mean(axis=1)
    log_vol = np.log(mkt.rolling(21).std() * np.sqrt(252))

    z = returns / returns.rolling(63).std()
    zbar = z.mean(axis=1)
    var_bar = zbar.rolling(63).var()
    avg_corr = ((N * var_bar - 1.0) / (N - 1.0)).clip(-0.1, 1.0)

    disp = returns.std(axis=1)
    log_disp = np.log(disp.rolling(21).mean())
    return pd.DataFrame(
        {"log_vol": log_vol, "avg_corr": avg_corr, "log_disp": log_disp}
    ).dropna()


class GaussianHMM:
    """Minimal diagonal-covariance Gaussian HMM (Baum–Welch + filtering)."""

    def __init__(self, n_states: int = 2, n_iter: int = 200, tol: float = 1e-6,
                 min_var: float = 1e-4, seed: int = 0):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.min_var = min_var
        self.seed = seed

    # ---------------------------------------------------------- internals
    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - self.means_[None, :, :]
        return -0.5 * (
            np.sum(diff**2 / self.vars_[None], axis=2)
            + np.sum(np.log(2 * np.pi * self.vars_), axis=1)[None]
        )

    def _scaled_passes(self, logB: np.ndarray, backward: bool = True):
        """Scaled forward/backward (Rabiner, 1989). Returns alpha, beta, c, ll.

        ``alpha[t]`` is the filtered distribution P(s_t | x_1..x_t).
        """
        T, S = logB.shape
        shift = logB.max(axis=1, keepdims=True)
        B = np.exp(logB - shift)
        A = self.trans_
        alpha = np.empty((T, S))
        c = np.empty(T)
        a = self.start_ * B[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * B[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        ll = float(np.log(c).sum() + shift.sum())
        beta = None
        if backward:
            beta = np.ones((T, S))
            for t in range(T - 2, -1, -1):
                beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        return alpha, beta, c, B, ll

    # ------------------------------------------------------------- public
    def fit(self, X: np.ndarray) -> "GaussianHMM":
        X = np.asarray(X, dtype=float)
        T, D = X.shape
        S = self.n_states
        # Initialize by quantiles of the first feature (volatility).
        order = np.argsort(X[:, 0])
        chunks = np.array_split(order, S)
        self.means_ = np.array([X[c].mean(axis=0) for c in chunks])
        self.vars_ = np.array([X[c].var(axis=0) + self.min_var for c in chunks])
        self.start_ = np.full(S, 1.0 / S)
        self.trans_ = np.full((S, S), 0.02 / max(S - 1, 1))
        np.fill_diagonal(self.trans_, 0.98)

        prev = -np.inf
        for _ in range(self.n_iter):
            logB = self._log_emission(X)
            alpha, beta, c, B, ll = self._scaled_passes(logB)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = (
                alpha[:-1, :, None] * self.trans_[None]
                * (B[1:] * beta[1:])[:, None, :] / c[1:, None, None]
            ).sum(axis=0)

            self.start_ = gamma[0] / gamma[0].sum()
            self.trans_ = xi / xi.sum(axis=1, keepdims=True)
            w = gamma.sum(axis=0)
            self.means_ = (gamma.T @ X) / w[:, None]
            self.vars_ = (gamma.T @ X**2) / w[:, None] - self.means_**2
            self.vars_ = np.maximum(self.vars_, self.min_var)
            if ll - prev < self.tol:
                break
            prev = ll

        # Order states by mean of the first feature (low vol first).
        o = np.argsort(self.means_[:, 0])
        self.means_, self.vars_ = self.means_[o], self.vars_[o]
        self.start_ = self.start_[o]
        self.trans_ = self.trans_[np.ix_(o, o)]
        self.loglik_ = prev
        return self

    def filter(self, X: np.ndarray) -> np.ndarray:
        """Filtered state probabilities P(s_t | x_1..x_t)."""
        logB = self._log_emission(np.asarray(X, dtype=float))
        alpha, *_ = self._scaled_passes(logB, backward=False)
        return alpha


def walk_forward_regimes(
    features: pd.DataFrame,
    first_fit: str | pd.Timestamp,
    n_states: int = 2,
    min_train: int = 252,
) -> tuple[pd.DataFrame, list[dict]]:
    """Annually refit HMM; return filtered probabilities with no look-ahead.

    For each calendar year ``Y`` from ``first_fit`` on, the HMM is fit
    on all feature rows strictly before 1 January ``Y`` (standardized
    with that training set's mean / std) and the forward filter is run
    over the full history up to the end of ``Y``; only year ``Y``'s
    probabilities are kept.
    """
    first_fit = pd.Timestamp(first_fit)
    years = sorted({d.year for d in features.index if d >= first_fit})
    out, fits = [], []
    for Y in years:
        cut = pd.Timestamp(f"{Y}-01-01")
        train = features[features.index < cut]
        if len(train) < min_train:
            continue
        mu, sd = train.mean(), train.std()
        hmm = GaussianHMM(n_states=n_states).fit(((train - mu) / sd).values)
        hist = features[features.index < pd.Timestamp(f"{Y + 1}-01-01")]
        probs = hmm.filter(((hist - mu) / sd).values)
        p = pd.DataFrame(
            probs, index=hist.index, columns=[f"p_state{s}" for s in range(n_states)]
        )
        out.append(p[p.index >= cut])
        fits.append(
            {
                "year": Y,
                "means": hmm.means_ * sd.values + mu.values,
                "trans": hmm.trans_,
                "loglik": hmm.loglik_,
            }
        )
    probs = pd.concat(out)
    probs["p_stress"] = probs[f"p_state{n_states - 1}"]
    return probs, fits
