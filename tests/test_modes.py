"""Sanity tests for the normal-mode decomposition and backtest pipeline.

Run with:

    python -m pytest tests/
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.backtest import walk_forward_backtest
from src.modes import (
    clean_covariance,
    decompose,
    marchenko_pastur_bounds,
)
from src.strategy import build_positions, mode_signals


rng = np.random.default_rng(42)


# --------------------------------------------------------------- MP formula


def test_marchenko_pastur_bounds_basic():
    """MP bounds satisfy known limits."""
    # q = 0: bounds collapse to sigma^2.
    lo, hi = marchenko_pastur_bounds(T=10_000_000, N=1, sigma2=1.0)
    assert lo == pytest.approx(1.0, abs=1e-3)
    assert hi == pytest.approx(1.0, abs=1e-3)

    # q = 1: lower bound = 0, upper bound = 4 sigma^2.
    lo, hi = marchenko_pastur_bounds(T=100, N=100, sigma2=1.0)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(4.0, abs=1e-9)


def test_marchenko_pastur_bounds_scaling():
    """Bounds scale linearly with sigma^2."""
    lo1, hi1 = marchenko_pastur_bounds(T=1000, N=100, sigma2=1.0)
    lo2, hi2 = marchenko_pastur_bounds(T=1000, N=100, sigma2=2.0)
    assert lo2 == pytest.approx(2.0 * lo1)
    assert hi2 == pytest.approx(2.0 * hi1)


def test_marchenko_pastur_rejects_bad_input():
    with pytest.raises(ValueError):
        marchenko_pastur_bounds(T=0, N=10)
    with pytest.raises(ValueError):
        marchenko_pastur_bounds(T=10, N=0)


# ---------------------------------------------------------- decomposition


def _iid_returns(T: int, N: int, seed: int = 0) -> pd.DataFrame:
    r = np.random.default_rng(seed).standard_normal((T, N)) * 0.01
    idx = pd.date_range("2020-01-01", periods=T, freq="B")
    cols = [f"A{i}" for i in range(N)]
    return pd.DataFrame(r, index=idx, columns=cols)


def test_decompose_orthonormal_eigenvectors():
    returns = _iid_returns(500, 30, seed=1)
    modes = decompose(returns)
    V = modes.eigenvectors
    assert np.allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-9)


def test_decompose_eigenvalues_descending():
    returns = _iid_returns(500, 30, seed=2)
    modes = decompose(returns)
    diffs = np.diff(modes.eigenvalues)
    assert (diffs <= 1e-9).all()


def test_decompose_iid_no_signal_modes():
    """For iid Gaussian returns, no eigenvalues should exceed the MP bound
    beyond a small number of outliers (finite-sample fluctuations)."""
    T, N = 2000, 50
    returns = _iid_returns(T, N, seed=3)
    modes = decompose(returns)
    # With T=2000, N=50, the finite-sample MP bulk is tight. Expect at
    # most a handful of outliers.
    assert modes.n_signal_modes <= 5, (
        f"Expected ~0 signal modes for iid returns, got {modes.n_signal_modes}."
    )


def test_decompose_planted_factor_produces_signal_mode():
    """Adding a common factor should produce one clear signal mode."""
    T, N = 1000, 40
    idio = np.random.default_rng(4).standard_normal((T, N)) * 0.01
    factor = np.random.default_rng(5).standard_normal(T) * 0.03
    loadings = np.random.default_rng(6).uniform(0.5, 1.5, size=N)
    R = idio + np.outer(factor, loadings)
    idx = pd.date_range("2020-01-01", periods=T, freq="B")
    returns = pd.DataFrame(R, index=idx, columns=[f"A{i}" for i in range(N)])

    modes = decompose(returns)
    assert modes.n_signal_modes >= 1
    # Top eigenvalue should dwarf the MP bound.
    assert modes.eigenvalues[0] > 5 * modes.mp_upper


def test_clean_covariance_preserves_trace():
    returns = _iid_returns(1000, 50, seed=7)
    modes = decompose(returns)
    C_clean = clean_covariance(modes)
    # Trace = sum of eigenvalues, invariant under our cleaning rule.
    assert np.trace(C_clean) == pytest.approx(modes.eigenvalues.sum(), rel=1e-9)


# ---------------------------------------------------------------- strategy


def test_mode_signals_are_negative_zscore():
    """Signal should be -zscore: a large positive mode return today
    (relative to trailing window) implies a negative signal."""
    T = 100
    idx = pd.date_range("2020-01-01", periods=T, freq="B")
    r = pd.DataFrame({"mode_0": np.ones(T) * 0.001}, index=idx)
    r.iloc[-1, 0] = 1.0  # huge positive spike

    signals = mode_signals(r, lookback=20)
    # Last signal should be very negative.
    assert signals.iloc[-1, 0] < -3.0


def test_build_positions_unit_gross_exposure():
    T, N = 100, 20
    returns = _iid_returns(T + 300, N, seed=8)
    modes = decompose(returns.iloc[:300])
    # Force at least 3 signal modes for the test even on iid data.
    modes.n_signal_modes = max(modes.n_signal_modes, 3)

    mode_ret = modes.project(returns.iloc[300 - 20 :])
    signals = mode_signals(mode_ret, lookback=20)
    signals = signals.iloc[20:]  # drop warmup

    if len(signals) == 0 or modes.n_signal_modes <= 1:
        pytest.skip("Not enough signal modes for this seed.")

    pos = build_positions(modes, signals, market_mode_skip=1, max_modes=3)
    gross = pos.abs().sum(axis=1)
    # Rows with any nonzero position should have unit gross exposure.
    nonzero = gross[gross > 0]
    assert np.allclose(nonzero.values, 1.0, atol=1e-9)


# ---------------------------------------------------------------- backtest


def test_backtest_smoke():
    """Backtest runs end-to-end on synthetic data with a planted factor."""
    T, N = 800, 40
    idio = np.random.default_rng(9).standard_normal((T, N)) * 0.01
    factor = np.random.default_rng(10).standard_normal(T) * 0.02
    loadings = np.random.default_rng(11).uniform(0.5, 1.5, size=N)
    R = idio + np.outer(factor, loadings)
    idx = pd.date_range("2018-01-01", periods=T, freq="B")
    returns = pd.DataFrame(R, index=idx, columns=[f"A{i}" for i in range(N)])

    result = walk_forward_backtest(
        returns,
        estimation_window=252,
        reestimation_freq=42,
        signal_lookback=15,
        market_mode_skip=1,
        max_modes=10,
        transaction_cost_bps=1.0,
    )
    assert len(result.pnl) == len(returns)
    assert "sharpe" in result.metrics
    assert result.metrics["n_days"] > 0
