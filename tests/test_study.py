"""Tests for the episode engine, regime model, ML layer, and portfolio.

The most important property of every component is the absence of
look-ahead; several tests perturb the future and check that nothing
computed for the past changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.episodes import (
    FEATURES_MODE,
    ar1_half_life,
    build_episodes,
    reversion_positions,
    variance_ratio,
)
from src.mode_backtest import evaluate, mode_portfolio, newey_west_tstat
from src.regime import GaussianHMM, market_features, walk_forward_regimes
from src.selection import select_top_fraction, walk_forward_predict


def factor_returns(T=700, N=40, k=3, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.normal(0, 1, (N, k))
    F = rng.normal(0, 0.01, (T, k))
    E = rng.normal(0, 0.01, (T, N))
    idx = pd.bdate_range("2015-01-01", periods=T)
    return pd.DataFrame(F @ B.T + E, index=idx, columns=[f"A{i}" for i in range(N)])


# ------------------------------------------------------------- signals


def test_reversion_positions_use_only_past():
    rng = np.random.default_rng(0)
    seg = rng.normal(size=(80, 4))
    pos = reversion_positions(seg, lookback=60)
    seg2 = seg.copy()
    seg2[70:] += 100.0  # perturb trading days 10.. onwards
    pos2 = reversion_positions(seg2, lookback=60)
    # Position for trading day j uses rows < 60 + j, so days 0..10 are unchanged.
    np.testing.assert_allclose(pos[:11], pos2[:11])
    assert not np.allclose(pos[11:], pos2[11:])


def test_reversion_positions_sign():
    # A mode whose price has just jumped up should be shorted.
    seg = np.zeros((61, 1))
    seg[:59] = np.random.default_rng(1).normal(0, 0.01, (59, 1))
    seg[59] = 1.0
    assert reversion_positions(seg, 60)[0, 0] < 0
    assert reversion_positions(seg, 60, signal="return")[0, 0] < 0


def test_half_life_recovers_ar1():
    rng = np.random.default_rng(2)
    b, T = 0.9, 20_000
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = b * x[t - 1] + rng.normal()
    hl = ar1_half_life(x[:, None])[0]
    assert hl == pytest.approx(np.log(2) / -np.log(b), rel=0.1)
    # A random walk has no detectable reversion -> large half-life.
    rw = np.cumsum(rng.normal(size=(5000, 1)), axis=0)
    assert ar1_half_life(rw)[0] > 50


def test_variance_ratio_white_noise():
    r = np.random.default_rng(3).normal(size=(50_000, 1))
    assert variance_ratio(r, 5)[0] == pytest.approx(1.0, abs=0.03)


# ------------------------------------------------------------- episodes


def test_episodes_features_are_backward_looking():
    R = factor_returns()
    eps, panel, _ = build_episodes(R, window=252, hold=21, lookback=60, n_candidates=10)
    R2 = R.copy()
    cut = eps[5].start
    R2.iloc[cut:] = R2.iloc[cut:] * 3 + 0.01
    _, panel2, _ = build_episodes(R2, window=252, hold=21, lookback=60, n_candidates=10)
    feats = [f for f in FEATURES_MODE if f != "stability"] + ["absorption"]
    a = panel[panel.episode == 5][feats].values
    b = panel2[panel2.episode == 5][feats].values
    np.testing.assert_allclose(a, b)
    # ...but its label does depend on the future.
    assert not np.allclose(panel[panel.episode == 5].fwd_pnl, panel2[panel2.episode == 5].fwd_pnl)


def test_episodes_tile_the_sample():
    R = factor_returns(T=600)
    eps, panel, _ = build_episodes(R, window=252, hold=21, lookback=60, n_candidates=10)
    assert eps[0].start == 252 and eps[-1].end == len(R)
    for a, b in zip(eps, eps[1:]):
        assert a.end == b.start
    assert set(panel["mode"]) == set(range(1, 11))


# ------------------------------------------------------------- portfolio


def test_mode_portfolio_unit_gross():
    R = factor_returns()
    eps, panel, _ = build_episodes(R, n_candidates=10)
    w = pd.DataFrame({"episode": panel.episode, "mode": panel["mode"], "w": 1.0})
    pos = mode_portfolio(eps, w, R)
    gross = pos.iloc[eps[0].start :].abs().sum(axis=1)
    np.testing.assert_allclose(gross[gross > 0], 1.0)
    assert (pos.iloc[: eps[0].start] == 0).all().all()
    res = evaluate(pos, R, cost_bps=0)
    manual = (pos * R).sum(axis=1)
    np.testing.assert_allclose(res["pnl"], manual)


def test_mode_portfolio_single_mode_matches_standalone_sign():
    """Trading one mode reproduces the sign of its standalone PnL."""
    R = factor_returns()
    eps, panel, _ = build_episodes(R, n_candidates=10)
    w = panel.assign(w=(panel["mode"] == 3).astype(float))[["episode", "mode", "w"]]
    pos = mode_portfolio(eps, w, R)
    ep = eps[3]
    pnl = (pos * R).sum(axis=1).iloc[ep.start : ep.end].values
    standalone = ep.positions[:, 3] * ep.mode_returns[:, 3]
    np.testing.assert_array_equal(np.sign(pnl), np.sign(standalone))


def test_newey_west_iid():
    x = np.random.default_rng(4).normal(0.1, 1, 10_000)
    t = newey_west_tstat(x, 5)
    t_iid = x.mean() / (x.std() / np.sqrt(len(x)))
    assert t == pytest.approx(t_iid, rel=0.05)


# ------------------------------------------------------------- regimes


def test_hmm_recovers_two_states():
    rng = np.random.default_rng(5)
    states = np.repeat([0, 1, 0, 1], 500)
    X = np.where(states[:, None] == 0, rng.normal(0, 1, (2000, 2)), rng.normal(3, 1, (2000, 2)))
    hmm = GaussianHMM(2).fit(X)
    p = hmm.filter(X)
    acc = ((p[:, 1] > 0.5) == (states == 1)).mean()
    assert acc > 0.97
    assert hmm.means_[0, 0] < hmm.means_[1, 0]


def test_hmm_filter_is_causal():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(400, 2))
    hmm = GaussianHMM(2).fit(X)
    X2 = X.copy()
    X2[300:] += 5
    np.testing.assert_allclose(hmm.filter(X)[:300], hmm.filter(X2)[:300])


def test_walk_forward_regimes_no_lookahead():
    R = factor_returns(T=1200, N=30)
    f = market_features(R)
    p1, _ = walk_forward_regimes(f, first_fit="2017-01-01")
    f2 = f.copy()
    f2.loc["2018-06-01":] += 2.0
    p2, _ = walk_forward_regimes(f2, first_fit="2017-01-01")
    a = p1.loc[:"2018-05-31", "p_stress"]
    np.testing.assert_allclose(a, p2.loc[a.index, "p_stress"])


# ------------------------------------------------------------- selection


def test_walk_forward_predict_ignores_future_labels():
    rng = np.random.default_rng(7)
    n_ep, k = 40, 10
    panel = pd.DataFrame({
        "episode": np.repeat(np.arange(n_ep), k),
        "x1": rng.normal(size=n_ep * k),
        "x2": rng.normal(size=n_ep * k),
    })
    panel["y"] = (panel.x1 + rng.normal(size=len(panel)) > 0).astype(int)
    p1 = walk_forward_predict(panel, ["x1", "x2"], "logit", min_train_episodes=10)
    panel2 = panel.copy()
    panel2.loc[panel2.episode >= 30, "y"] = 1 - panel2.loc[panel2.episode >= 30, "y"]
    p2 = walk_forward_predict(panel2, ["x1", "x2"], "logit", min_train_episodes=10)
    m = panel.episode <= 30  # episode 30's model is trained on episodes < 30
    np.testing.assert_allclose(p1[m].dropna(), p2[m].dropna())
    assert p1[panel.episode < 10].isna().all()


def test_select_top_fraction():
    panel = pd.DataFrame({"episode": np.repeat([0, 1], 9), "s": np.tile(np.arange(9.0), 2)})
    w = select_top_fraction(panel, "s", 1 / 3)
    assert w.sum() == 6
    assert set(panel.loc[w == 1, "s"]) == {6.0, 7.0, 8.0}
