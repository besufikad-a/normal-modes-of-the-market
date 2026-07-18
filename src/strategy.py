"""Mode-reversion statistical arbitrage strategy.

Idea
----
Under the coupled-oscillator analogy, each normal mode of the return
covariance is a portfolio whose return series has variance equal to its
eigenvalue. Three regimes:

* **Top modes** (eigenvalue ≫ MP upper): carry systematic risk (market,
  broad sectors). We do not want to bet on their direction.
* **Noise modes** (eigenvalue ≤ MP upper): dominated by sampling noise;
  their eigenvectors are effectively random and not tradeable.
* **Signal modes** (between): statistically non-trivial linear
  combinations that carry limited variance. Under the null of a
  stationary market, cumulative returns of these mode portfolios should
  mean-revert.

The strategy z-scores each signal mode's return over a rolling window
and takes a position proportional to the negative z-score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .modes import NormalModes


def mode_signals(
    mode_returns: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """Compute mean-reversion signals from mode returns.

    Signal for mode ``i`` on day ``t`` is the negative rolling z-score
    of the mode's return over the trailing ``lookback`` days:

        s_i(t) = -(r_i(t) - μ_i(t)) / σ_i(t)

    Positive signal = mode has underperformed = bet on reversion up (long
    the mode portfolio). Negative signal = short.

    Parameters
    ----------
    mode_returns : pd.DataFrame
        Output of ``NormalModes.project``. One column per mode.
    lookback : int
        Rolling window length.
    """
    rolling_mean = mode_returns.rolling(lookback).mean()
    rolling_std = mode_returns.rolling(lookback).std()
    # Avoid divide-by-zero for degenerate windows.
    rolling_std = rolling_std.replace(0.0, np.nan)
    z = (mode_returns - rolling_mean) / rolling_std
    return -z


def build_positions(
    modes: NormalModes,
    signals: pd.DataFrame,
    market_mode_skip: int = 1,
    max_modes: int | None = None,
) -> pd.DataFrame:
    """Convert mode-level signals into per-asset positions.

    Positions are the signal-weighted sum of the mode eigenvectors
    (which are portfolios). Because eigenvectors are orthogonal, the
    resulting asset-level positions are naturally decorrelated across
    modes.

    Parameters
    ----------
    modes : NormalModes
        The decomposition. Only signal modes (eigenvalue > MP upper)
        are considered eligible.
    signals : pd.DataFrame
        Mode-level signals from ``mode_signals``, one column per mode,
        aligned with ``modes.eigenvectors``.
    market_mode_skip : int, default 1
        Number of top signal modes to exclude. Skipping the market mode
        (typically mode 0) makes the strategy approximately market-
        neutral by construction.
    max_modes : int or None, default None
        Cap on total signal modes considered. Trading modes just above
        the MP floor can be unstable because their eigenvectors are
        sensitive to sample noise; a cap improves realism.

    Returns
    -------
    pd.DataFrame
        Rows are dates, columns are tickers. Positions are rescaled per
        day to unit gross exposure (∑|w_i| = 1).
    """
    n_signal = modes.n_signal_modes
    if max_modes is not None:
        n_signal = min(n_signal, max_modes)

    used = list(range(market_mode_skip, n_signal))
    if not used:
        raise ValueError(
            f"No modes available after skipping {market_mode_skip} and "
            f"capping at {max_modes}. n_signal_modes={modes.n_signal_modes}."
        )

    # Signal matrix on the used modes: (T, k)
    S = signals.iloc[:, used].fillna(0.0).values
    # Eigenvectors for used modes: (N, k)
    V = modes.eigenvectors[:, used]

    # Asset-level positions: (T, N) = signal-weighted sum of mode portfolios
    positions = S @ V.T

    # Rescale to unit gross exposure per day.
    gross = np.abs(positions).sum(axis=1, keepdims=True)
    gross = np.where(gross == 0, 1.0, gross)
    positions = positions / gross

    return pd.DataFrame(
        positions,
        index=signals.index,
        columns=modes.tickers,
    )
