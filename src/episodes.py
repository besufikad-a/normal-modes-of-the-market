"""Walk-forward mode episodes: decomposition, signals, features, labels.

An **episode** is one re-estimation step of the walk-forward. At
re-estimation day ``t`` the correlation matrix is estimated on the
trailing window ``[t - W, t)``, the modes are frozen, and they are
traded over the holding period ``[t, t + H)``. Everything an episode
records about the window (features) uses data strictly before ``t``;
everything it records about the holding period (labels, PnL) uses data
from ``t`` onwards. Because the holding period equals the
re-estimation step, episode ``i``'s label is fully realized before
episode ``i + 1`` starts, which is what lets the ML layer train on
episodes ``< j`` when predicting episode ``j`` without leakage.

Mode coordinates
----------------
The modes are eigenvectors ``v_k`` of the *correlation* matrix, so the
natural mode return is the projection of volatility-scaled returns,

    r_k(d) = Σ_i v_ik r_i(d) / σ_i,

with ``σ_i`` the in-window volatility of asset ``i``. The tradable
asset weights of mode ``k`` are therefore ``v_ik / σ_i`` (as in
Avellaneda & Lee, 2010). In-window, ``Var(r_k) ≈ λ_k``.

Signal
------
Each mode has a cumulative "price" ``X_k(d) = Σ_{s ≤ d} r_k(s)``. The
reversion signal on day ``d`` is the negative z-score of ``X_k(d-1)``
relative to its trailing ``L``-day mean and standard deviation, clipped
to ``±clip``. This is a bet that the *level* of the mode portfolio
reverts, which is the coupled-oscillator analogue (a displaced
coordinate is pulled back toward equilibrium). The original repo
specification — a z-score of the *daily return* — is available as
``signal="return"`` for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .modes import decompose


# --------------------------------------------------------------- signals


def reversion_positions(
    seg: np.ndarray, lookback: int, signal: str = "level", clip: float = 3.0
) -> np.ndarray:
    """Mode positions for each trading day of a segment.

    Parameters
    ----------
    seg : np.ndarray, shape (lookback + H, K)
        Mode returns. The first ``lookback`` rows are warm-up history;
        the remaining ``H`` rows are the trading days.
    lookback : int
        Trailing window ``L`` for the z-score.
    signal : {"level", "return"}
        ``"level"`` z-scores the cumulative mode price; ``"return"``
        z-scores the daily mode return (original repo specification).
    clip : float
        Absolute cap on the z-score.

    Returns
    -------
    np.ndarray, shape (H, K)
        Position held over trading day ``j``, using information through
        the close of the previous day only.
    """
    H = seg.shape[0] - lookback
    if H <= 0:
        raise ValueError("Segment must be longer than the lookback.")
    base = np.cumsum(seg, axis=0) if signal == "level" else seg
    if signal not in ("level", "return"):
        raise ValueError(f"Unknown signal {signal!r}.")
    # windows[j] covers rows j .. j+L-1, i.e. the L days before trading day j.
    windows = sliding_window_view(base[: lookback + H - 1], lookback, axis=0)
    mean = windows.mean(axis=-1)
    std = windows.std(axis=-1, ddof=1)
    last = windows[..., -1]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(std > 0, (last - mean) / std, 0.0)
    return -np.clip(z, -clip, clip)


def ar1_half_life(x: np.ndarray, cap: float = 252.0) -> np.ndarray:
    """OU half-life (days) of each column of a level series via AR(1).

    Fits ``x_{s+1} = a + b x_s + e`` by OLS. The half-life is
    ``ln 2 / (-ln b)`` for ``0 < b < 1`` and ``cap`` otherwise (no
    detectable reversion).
    """
    x0, x1 = x[:-1], x[1:]
    x0c = x0 - x0.mean(axis=0)
    x1c = x1 - x1.mean(axis=0)
    b = (x0c * x1c).sum(axis=0) / (x0c**2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        hl = np.where((b > 0) & (b < 1), np.log(2) / -np.log(b), cap)
    return np.clip(hl, 0.5, cap)


def lag1_autocorr(r: np.ndarray) -> np.ndarray:
    """Lag-1 autocorrelation of each column."""
    rc = r - r.mean(axis=0)
    return (rc[1:] * rc[:-1]).sum(axis=0) / (rc**2).sum(axis=0)


def variance_ratio(r: np.ndarray, q: int = 5) -> np.ndarray:
    """Lo–MacKinlay variance ratio VR(q) of each column (overlapping sums)."""
    rc = r - r.mean(axis=0)
    rq = sliding_window_view(rc, q, axis=0).sum(axis=-1)
    return rq.var(axis=0, ddof=1) / (q * rc.var(axis=0, ddof=1))


# --------------------------------------------------------------- episodes


@dataclass
class Episode:
    """One re-estimation step of the walk-forward."""

    idx: int
    start: int  # first trading row (inclusive)
    end: int  # last trading row (exclusive)
    date: pd.Timestamp
    eigenvalues: np.ndarray  # (N,) all eigenvalues, descending
    V: np.ndarray  # (N, K+1) eigenvectors for modes 0..K
    sigma: np.ndarray  # (N,) in-window asset vols
    mp_upper: float
    n_signal: int
    positions: np.ndarray  # (H, K+1) standalone mode positions, trading days
    mode_returns: np.ndarray  # (H, K+1) mode returns on trading days
    features: pd.DataFrame = field(repr=False, default=None)


FEATURES_MODE = [
    "mode_index",
    "lam_over_mp",
    "log_lambda",
    "eig_gap",
    "log_half_life",
    "ac1",
    "vr5",
    "stability",
    "vol_ratio",
    "participation",
    "max_weight",
    "trail_rev_sharpe",
]
FEATURES_MARKET = ["absorption"]


def build_episodes(
    returns: pd.DataFrame,
    window: int = 252,
    hold: int = 21,
    lookback: int = 60,
    n_candidates: int | None = 30,
    signal: str = "level",
    clip: float = 3.0,
    trail: int = 63,
    delay: int = 0,
    collect_all_modes: bool = False,
    verbose: bool = False,
) -> tuple[list[Episode], pd.DataFrame, dict | None]:
    """Run the walk-forward decomposition and build the mode panel.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns, NaN-free.
    window, hold, lookback : int
        Estimation window ``W``, holding period / re-estimation step
        ``H``, and signal lookback ``L``.
    n_candidates : int or None
        Candidate modes are ``1 .. n_candidates`` (mode 0, the market,
        is always excluded from the candidates but kept in ``V``).
        ``None`` uses every non-degenerate mode (rank ``min(W-1, N)``).
    delay : int
        Extra days of execution delay: the position for day ``d`` uses
        information through ``d - 1 - delay``.
    collect_all_modes : bool
        Also record standalone forward reversion PnL for *every*
        non-degenerate mode (used for the spectrum-wide test).

    Returns
    -------
    episodes : list[Episode]
    panel : pd.DataFrame
        One row per (episode, candidate mode) with features and labels.
    spectrum : dict or None
        If ``collect_all_modes``: per-episode standalone daily PnL of
        every mode, normalized by ``sqrt(λ_k)``.
    """
    if lookback > window:
        raise ValueError("lookback must not exceed the estimation window.")
    R = returns.values
    T, N = R.shape
    rank = min(window - 1, N)
    K = rank - 2 if n_candidates is None else min(n_candidates, rank - 2)
    episodes: list[Episode] = []
    rows: list[pd.DataFrame] = []
    spec_pnl, spec_ret, spec_dates, spec_nsig = [], [], [], []
    V_prev = None

    start = window
    while start < T:
        end = min(start + hold, T)
        H = end - start
        win = returns.iloc[start - window : start]
        modes = decompose(win, standardize=True)
        sigma = win.std(ddof=1).values
        lam = modes.eigenvalues
        V_all = modes.eigenvectors

        n_modes = rank if collect_all_modes else K + 2
        V = V_all[:, :n_modes]
        Z = R[start - window : end] / sigma  # (W + H, N)
        M = Z @ V  # mode returns, (W + H, n_modes)

        pos = reversion_positions(
            M[window - lookback - delay : end - start + window - delay],
            lookback, signal, clip,
        )
        fwd = M[window:]
        pnl = pos * fwd  # (H, n_modes)

        if collect_all_modes:
            scale = np.sqrt(lam[:n_modes])
            spec_pnl.append(pnl / scale)
            spec_ret.append(fwd / scale)
            spec_dates.append(returns.index[start:end])
            spec_nsig.append(modes.n_signal_modes)

        Kc = K + 1
        Mw = M[:window, :Kc]  # in-window mode returns
        Vc = V[:, :Kc]

        # ---- features for candidate modes 1..K (window data only)
        ks = np.arange(1, Kc)
        lam_c = lam[ks]
        gap_up = lam[ks - 1] - lam_c
        gap_dn = lam_c - lam[ks + 1]
        eig_gap = np.minimum(gap_up, gap_dn) / lam_c

        hl = ar1_half_life(np.cumsum(Mw[:, ks], axis=0))
        ac1 = lag1_autocorr(Mw[:, ks])
        vr5 = variance_ratio(Mw[:, ks], 5)
        if V_prev is None:
            stab = np.full(K, np.nan)
        else:
            # Best overlap with any of the previous episode's leading modes.
            stab = np.abs(Vc[:, ks].T @ V_prev).max(axis=1)
        vol_ratio = Mw[-21:, ks].std(axis=0, ddof=1) / np.sqrt(lam_c)
        ipr = (Vc[:, ks] ** 4).sum(axis=0)
        participation = 1.0 / (ipr * N)  # effective fraction of assets used
        w_assets = np.abs(Vc[:, ks] / sigma[:, None])
        max_weight = w_assets.max(axis=0) / w_assets.sum(axis=0)

        trail_seg = M[window - trail - lookback : window, :Kc][:, ks]
        tpos = reversion_positions(trail_seg, lookback, signal, clip)
        tpnl = tpos * trail_seg[lookback:]
        trail_sharpe = tpnl.mean(axis=0) / tpnl.std(axis=0, ddof=1) * np.sqrt(252)

        fwd_pnl = pnl[:, ks].sum(axis=0)
        fwd_ret = fwd[:, ks]
        fwd_ac1 = lag1_autocorr(M[window - 1 :, ks]) if H > 2 else np.full(K, np.nan)

        ep_idx = len(episodes)
        feats = pd.DataFrame(
            {
                "episode": ep_idx,
                "date": returns.index[start],
                "mode": ks,
                "mode_index": ks,
                "above_mp": lam_c > modes.mp_upper,
                "lam_over_mp": lam_c / modes.mp_upper,
                "log_lambda": np.log(lam_c),
                "eig_gap": eig_gap,
                "log_half_life": np.log(hl),
                "ac1": ac1,
                "vr5": vr5,
                "stability": stab,
                "vol_ratio": vol_ratio,
                "participation": participation,
                "max_weight": max_weight,
                "trail_rev_sharpe": trail_sharpe,
                "absorption": lam[0] / N,
                "n_signal": modes.n_signal_modes,
                "fwd_pnl": fwd_pnl,
                "fwd_pnl_norm": fwd_pnl / np.sqrt(lam_c),
                "fwd_ac1": fwd_ac1,
                "fwd_vol": fwd_ret.std(axis=0, ddof=1) if H > 1 else np.nan,
                "y": (fwd_pnl > 0).astype(int),
                "hold_days": H,
            }
        )
        rows.append(feats)

        episodes.append(
            Episode(
                idx=ep_idx,
                start=start,
                end=end,
                date=returns.index[start],
                eigenvalues=lam,
                V=Vc,
                sigma=sigma,
                mp_upper=modes.mp_upper,
                n_signal=modes.n_signal_modes,
                positions=pos[:, :Kc],
                mode_returns=fwd[:, :Kc],
            )
        )
        V_prev = Vc
        if verbose and ep_idx % 24 == 0:
            print(
                f"[episodes] {ep_idx:4d} {returns.index[start].date()} "
                f"n_signal={modes.n_signal_modes}"
            )
        start = end

    panel = pd.concat(rows, ignore_index=True)
    spectrum = None
    if collect_all_modes:
        n_min = min(p.shape[1] for p in spec_pnl)
        spectrum = {
            "pnl": np.vstack([p[:, :n_min] for p in spec_pnl]),
            "ret": np.vstack([p[:, :n_min] for p in spec_ret]),
            "dates": pd.DatetimeIndex(np.concatenate(spec_dates)),
            "n_signal": np.array(spec_nsig),
            "episode_of_day": np.concatenate(
                [np.full(len(d), i) for i, d in enumerate(spec_dates)]
            ),
        }
    return episodes, panel, spectrum
