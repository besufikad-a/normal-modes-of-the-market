"""Portfolio construction and PnL from pre-computed mode episodes.

Given per-(episode, mode) weights ``w_k`` (0 = do not trade, 1 = trade
at full size), the asset position on trading day ``d`` of an episode is

    x_i(d) = e · Σ_k w_k · s_k(d) / sqrt(λ_k) · v_ik / σ_i,

rescaled to unit gross exposure ``Σ_i |x_i| = 1`` before the optional
episode-level exposure multiplier ``e`` (the regime overlay). The
``1/sqrt(λ_k)`` factor gives every selected mode the same ex-ante
volatility (risk parity across modes). ``s_k(d)`` is the standalone
reversion position of mode ``k``, which already uses information only
through the close of day ``d - 1``, so ``PnL(d) = x(d) · r(d)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import compute_metrics
from .episodes import Episode


def mode_portfolio(
    episodes: list[Episode],
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    exposure: pd.Series | None = None,
) -> pd.DataFrame:
    """Daily asset positions implied by mode weights.

    Parameters
    ----------
    weights : pd.DataFrame
        Columns ``episode``, ``mode``, ``w``. Missing (episode, mode)
        pairs have weight 0.
    exposure : pd.Series or None
        Episode-indexed gross-exposure multiplier (default 1).
    """
    X = np.zeros(returns.shape)
    wmap = {
        e: g.set_index("mode")["w"] for e, g in weights[weights["w"] != 0].groupby("episode")
    }
    for ep in episodes:
        w_ep = wmap.get(ep.idx)
        if w_ep is None or len(w_ep) == 0:
            continue
        ks = w_ep.index.to_numpy()
        w = w_ep.to_numpy()
        P = ep.positions[:, ks] * (w / np.sqrt(ep.eigenvalues[ks]))[None]  # (H, k)
        x = (P @ ep.V[:, ks].T) / ep.sigma[None]  # (H, N)
        gross = np.abs(x).sum(axis=1, keepdims=True)
        x = np.divide(x, gross, out=np.zeros_like(x), where=gross > 0)
        if exposure is not None:
            x *= float(exposure.get(ep.idx, 1.0))
        X[ep.start : ep.end] = x
    return pd.DataFrame(X, index=returns.index, columns=returns.columns)


def evaluate(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 2.0,
    start: pd.Timestamp | None = None,
) -> dict:
    """PnL series and metrics for a position panel.

    Turnover is ``Σ_i |x_i(d) - x_i(d-1)|``; costs are ``cost_bps`` per
    unit of turnover. Metrics are computed from ``start`` on.
    """
    pnl_gross = (positions * returns).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    turnover.iloc[0] = positions.iloc[0].abs().sum()
    pnl = pnl_gross - turnover * cost_bps / 1e4
    mkt = returns.mean(axis=1)
    if start is not None:
        sl = slice(start, None)
        pnl, pnl_gross, turnover, mkt = pnl[sl], pnl_gross[sl], turnover[sl], mkt[sl]
    m = compute_metrics(pnl)
    mg = compute_metrics(pnl_gross)
    m["gross_sharpe"] = mg["sharpe"]
    m["gross_ann_return"] = mg["annualized_return"]
    m["avg_turnover"] = float(turnover.mean())
    m["avg_gross"] = float(positions.loc[pnl.index].abs().sum(axis=1).mean())
    m["market_corr"] = float(np.corrcoef(pnl_gross, mkt)[0, 1])
    cum = pnl.cumsum()
    m["calmar"] = m["annualized_return"] / abs(m["max_drawdown"]) if m["max_drawdown"] < 0 else np.nan
    return {"pnl": pnl, "pnl_gross": pnl_gross, "turnover": turnover, "metrics": m, "cum": cum}


def newey_west_tstat(x: np.ndarray, lags: int = 5) -> float:
    """t-statistic of the mean with Newey–West (Bartlett) standard errors."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    xc = x - x.mean()
    s = xc @ xc / n
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * (xc[l:] @ xc[:-l]) / n
    return float(x.mean() / np.sqrt(s / n))


def sharpe_diff_bootstrap(
    a: pd.Series, b: pd.Series, n_boot: int = 2000, block: int = 21, seed: int = 0
) -> tuple[float, float, float]:
    """Stationary-block bootstrap of the Sharpe difference ``SR(a) - SR(b)``.

    Returns (difference, 95% CI low, one-sided p-value for diff ≤ 0).
    """
    df = pd.concat([a, b], axis=1).dropna().values
    n = len(df)
    rng = np.random.default_rng(seed)

    def sr(x):
        s = x.std(axis=0, ddof=1)
        return x.mean(axis=0) / s * np.sqrt(252)

    base = sr(df)
    diff = base[0] - base[1]
    diffs = np.empty(n_boot)
    nb = int(np.ceil(n / block))
    for i in range(n_boot):
        starts = rng.integers(0, n - block, nb)
        idx = (starts[:, None] + np.arange(block)[None]).ravel()[:n]
        s = sr(df[idx])
        diffs[i] = s[0] - s[1]
    lo = np.percentile(diffs, 2.5)
    # Centered bootstrap p-value for H0: diff <= 0.
    p = float(np.mean(diffs - diff >= diff))
    return float(diff), float(lo), p
