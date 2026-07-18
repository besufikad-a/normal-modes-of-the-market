"""Walk-forward backtest for the normal-mode reversion strategy.

Design decisions
----------------
* **Rolling re-estimation.** The covariance matrix (and thus the modes)
  is re-estimated every ``reestimation_freq`` days on a trailing window
  of length ``estimation_window``. This avoids look-ahead in the
  eigendecomposition and lets the modes adapt to regime shifts.
* **Signal warmup.** After each re-estimation, the first
  ``signal_lookback`` days are used to build the z-score history and
  positions are held at zero.
* **PnL timing.** Positions are lagged by one day: today's signal
  drives tomorrow's return. This is the standard convention for daily
  data and avoids trivial look-ahead.
* **Costs.** A flat basis-point charge on daily gross turnover. Real
  markets have market-impact and financing costs that scale non-
  linearly; this is a placeholder.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .modes import decompose
from .strategy import build_positions, mode_signals


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    pnl: pd.Series
    pnl_gross: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series
    metrics: dict

    def summary(self) -> str:
        """Human-readable metrics table."""
        lines = ["Backtest summary", "-" * 44]
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k:24s} {v:>14.4f}")
            else:
                lines.append(f"  {k:24s} {v!s:>14}")
        return "\n".join(lines)


def compute_metrics(pnl: pd.Series, ann_factor: int = 252) -> dict:
    """Standard performance metrics.

    Parameters
    ----------
    pnl : pd.Series
        Daily PnL series (log-return-like: additive).
    ann_factor : int
        Annualization factor. 252 = US trading days.
    """
    pnl = pnl.dropna()
    if len(pnl) == 0:
        return {"n_days": 0}

    mean = pnl.mean() * ann_factor
    vol = pnl.std(ddof=1) * np.sqrt(ann_factor)
    sharpe = mean / vol if vol > 0 else float("nan")

    cum = pnl.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd = float(drawdown.min())

    hit_rate = float((pnl > 0).mean())

    return {
        "n_days": int(len(pnl)),
        "annualized_return": float(mean),
        "annualized_vol": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
    }


def walk_forward_backtest(
    returns: pd.DataFrame,
    estimation_window: int = 252,
    reestimation_freq: int = 21,
    signal_lookback: int = 20,
    market_mode_skip: int = 1,
    max_modes: int | None = 20,
    transaction_cost_bps: float = 1.0,
    verbose: bool = False,
) -> BacktestResult:
    """Walk-forward backtest of the mode-reversion strategy.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns. Rows = dates, columns = tickers. NaN-free.
    estimation_window : int, default 252
        Trailing window for covariance / mode estimation (~1 year).
    reestimation_freq : int, default 21
        Days between mode re-estimations (~1 month).
    signal_lookback : int, default 20
        Rolling window for the mean-reversion z-score.
    market_mode_skip : int, default 1
        Number of top modes to exclude (market-neutral if 1).
    max_modes : int or None, default 20
        Cap on total signal modes used. None = all signal modes.
    transaction_cost_bps : float, default 1.0
        Round-trip cost per unit turnover in basis points.
    verbose : bool
        Print progress during walk-forward.

    Returns
    -------
    BacktestResult
        PnL series, positions, turnover, and summary metrics.
    """
    dates = returns.index
    T = len(returns)

    if T < estimation_window + signal_lookback + 2:
        raise ValueError(
            f"Not enough data. Need ≥ {estimation_window + signal_lookback + 2} "
            f"days, got {T}."
        )

    positions = pd.DataFrame(0.0, index=dates, columns=returns.columns)

    start = estimation_window
    step_count = 0
    while start < T:
        end = min(start + reestimation_freq, T)
        window_returns = returns.iloc[start - estimation_window : start]

        try:
            modes = decompose(window_returns, standardize=True)
        except (np.linalg.LinAlgError, ValueError) as e:
            if verbose:
                print(f"[backtest] decompose failed at t={start}: {e}")
            start = end
            continue

        if modes.n_signal_modes <= market_mode_skip:
            if verbose:
                print(
                    f"[backtest] t={start}: only {modes.n_signal_modes} signal "
                    f"modes; skipping."
                )
            start = end
            continue

        # We need signal_lookback history of mode returns before the
        # trading slice begins.
        signal_slice = returns.iloc[start - signal_lookback : end]
        mode_ret = modes.project(signal_slice)
        signals = mode_signals(mode_ret, lookback=signal_lookback)

        # Only take positions during the trading slice.
        trading_signals = signals.iloc[signal_lookback:]
        if len(trading_signals) == 0:
            start = end
            continue

        pos = build_positions(
            modes,
            trading_signals,
            market_mode_skip=market_mode_skip,
            max_modes=max_modes,
        )
        # Reindex pos onto the full ticker set (missing tickers get 0).
        pos = pos.reindex(columns=returns.columns, fill_value=0.0)
        positions.loc[pos.index, :] = pos.values

        step_count += 1
        if verbose and step_count % 12 == 0:
            print(
                f"[backtest] step {step_count}: t={start}/{T}, "
                f"n_signal_modes={modes.n_signal_modes}"
            )
        start = end

    # PnL with 1-day lag.
    pos_lag = positions.shift(1).fillna(0.0)
    pnl_gross = (pos_lag * returns).sum(axis=1)

    turnover = (positions - positions.shift(1)).abs().sum(axis=1).fillna(0.0)
    cost = turnover * (transaction_cost_bps / 10_000.0)
    pnl_net = pnl_gross - cost

    metrics = compute_metrics(pnl_net)
    metrics["gross_sharpe"] = compute_metrics(pnl_gross).get("sharpe", float("nan"))
    metrics["avg_daily_turnover"] = float(turnover.mean())
    metrics["transaction_cost_bps"] = float(transaction_cost_bps)

    return BacktestResult(
        pnl=pnl_net,
        pnl_gross=pnl_gross,
        positions=positions,
        turnover=turnover,
        metrics=metrics,
    )
