"""Full study for the working paper *Normal Modes of the Market*.

Usage (from repo root)::

    python scripts/run_study.py            # main results + robustness
    python scripts/run_study.py --quick    # skip the robustness grid

Stages
------
1. Data: current S&P 500 constituents with complete 2010–2024 history.
2. Spectrum-wide test: out-of-sample standalone reversion of every mode,
   by eigenvalue rank, against the Marchenko–Pastur edge.
3. Regimes: walk-forward 2-state Gaussian HMM, filtered probabilities.
4. Mode-quality prediction: walk-forward logit / GBM / RF on two
   candidate pools (modes 1–30, pre-specified; full spectrum, added
   after stage 2).
5. Strategies: baselines vs. ML selection vs. ML + regime, on a common
   out-of-sample period, with placebo and bootstrap tests.
6. Robustness: costs, execution delay, window / lookback / holding
   period, selection fraction, model, sub-periods.

Writes figures to ``paper/figures``, LaTeX tables to ``paper/tables``,
and every number quoted in the paper to ``output/results.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import walk_forward_backtest
from src.data import compute_returns, download_prices, get_sp500_tickers
from src.episodes import FEATURES_MODE, build_episodes
from src.mode_backtest import (
    evaluate,
    mode_portfolio,
    newey_west_tstat,
    sharpe_diff_bootstrap,
)
from src.modes import decompose, marchenko_pastur_bounds
from src.regime import market_features, walk_forward_regimes
from src.selection import (
    block_bootstrap_auc,
    oos_auc,
    select_top_fraction,
    walk_forward_predict,
)

# ------------------------------------------------------------------ config

START, END = "2010-01-01", "2024-12-31"
CACHE = REPO_ROOT / "data" / "cache" / "sp500_prices_2010_2024.parquet"
FIG = REPO_ROOT / "paper" / "figures"
TAB = REPO_ROOT / "paper" / "tables"
OUT = REPO_ROOT / "output"

BASE = dict(window=252, hold=21, lookback=60)
MIN_TRAIN = 24  # episodes (~2 years) before the first prediction
FRAC = 1 / 3  # fraction of each pool selected
COST_BPS = 2.0
PRIMARY_MODEL = "logit"
REGIME_FEATURES = ["p_stress", "log_vol", "avg_corr", "absorption"]
FEATURES_ALL = FEATURES_MODE + REGIME_FEATURES
GROUPS = {
    "Eigenvalue separation": ["mode_index", "lam_over_mp", "log_lambda", "eig_gap"],
    "Half-life": ["log_half_life"],
    "Autocorrelation": ["ac1", "vr5"],
    "Eigenvector stability": ["stability"],
    "Volatility": ["vol_ratio"],
    "Concentration": ["participation", "max_weight"],
    "Trailing reversion": ["trail_rev_sharpe"],
    "Regime": REGIME_FEATURES,
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)
C = {"blue": "#2563eb", "orange": "#ea580c", "green": "#16a34a", "grey": "#6b7280",
     "red": "#dc2626", "purple": "#7c3aed"}

RESULTS: dict = {}
TEX_BUCKET = {
    'Market (k=0)': 'Market ($k=0$)',
    'Signal, above MP (1..n_s-1)': 'Signal, above MP ($1..n_s-1$)',
    'Edge bulk (n_s..n_s+24)': 'Edge bulk ($n_s..n_s+24$)',
    'Upper bulk (n_s+25..99)': 'Upper bulk ($n_s+25..99$)',
    'Middle bulk (100..174)': 'Middle bulk ($100..174$)',
    'Lower bulk (175..rank)': 'Lower bulk ($175..249$)',
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def fmt(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# ---------------------------------------------------------------- helpers


def attach_regimes(panel: pd.DataFrame, probs: pd.DataFrame, mfeat: pd.DataFrame) -> pd.DataFrame:
    """Attach the regime state as of the close *before* each episode."""
    reg = pd.concat([probs[["p_stress"]], mfeat[["log_vol", "avg_corr"]]], axis=1)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    rows = {}
    for d in dates:
        prior = reg[reg.index < d]
        rows[d] = prior.iloc[-1] if len(prior) else pd.Series(np.nan, index=reg.columns)
    reg_at = pd.DataFrame(rows).T
    reg_at.index.name = "date"
    out = panel.drop(columns=[c for c in reg.columns if c in panel], errors="ignore")
    return out.merge(reg_at, left_on="date", right_index=True, how="left")


def weights_frame(panel: pd.DataFrame, w: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"episode": panel["episode"], "mode": panel["mode"], "w": w})


def run_strategy(episodes, panel, w, returns, start, exposure=None, cost=COST_BPS):
    pos = mode_portfolio(episodes, weights_frame(panel, w), returns, exposure)
    res = evaluate(pos, returns, cost_bps=cost, start=start)
    res["positions"] = pos
    return res


def regime_exposure(pnl: pd.Series, panel: pd.DataFrame, episodes, min_stress: int = 6) -> pd.Series:
    """Walk-forward regime overlay.

    For episode j with filtered stress probability p_j (as of the close
    before the episode), exposure is

        e_j = (1 - p_j) * g_calm + p_j * g_stress,
        g_s = clip(SR_s / max(SR_calm, SR_stress), 0, 1),

    where SR_s is the strategy's own past Sharpe in regime s, using only
    days of episodes that ended before j started. The best regime so far
    is traded at full size, a worse one proportionally less, and a
    regime with negative past Sharpe not at all. If neither regime has
    a positive past Sharpe, nothing is traded. Until both regimes have
    at least ``min_stress`` past episodes and a year of history, e_j = 1.

    (A first version that could only de-risk in the *stress* state was
    never active, because the strategy performs better under stress;
    see the paper.)
    """
    ep_state = panel.groupby("episode")["p_stress"].first()
    day_ep = pd.Series(np.nan, index=pnl.index)
    for ep in episodes:
        day_ep.iloc[ep.start : ep.end] = ep.idx
    exp = {}
    for ep in episodes:
        j = ep.idx
        p = ep_state.get(j, np.nan)
        past = pnl[(day_ep < j) & pnl.notna()]
        past = past[past != 0]
        if np.isnan(p) or len(past) < 252:
            exp[j] = 1.0
            continue
        st = ep_state.reindex(day_ep[past.index].values).values > 0.5
        calm, stress = past[~st], past[st]
        n_c = len(set(day_ep[calm.index])) if len(calm) else 0
        n_s = len(set(day_ep[stress.index])) if len(stress) else 0
        if min(n_c, n_s) < min_stress:
            exp[j] = 1.0
            continue
        sr = np.array([calm.mean() / calm.std(), stress.mean() / stress.std()])
        best = sr.max()
        g = np.zeros(2) if best <= 0 else np.clip(sr / best, 0.0, 1.0)
        exp[j] = float((1 - p) * g[0] + p * g[1])
    return pd.Series(exp)


# ------------------------------------------------------------------ stages


def stage_data():
    log("Stage 1: data")
    tickers = get_sp500_tickers() if not CACHE.exists() else []
    prices = download_prices(tickers, start=START, end=END, cache_path=CACHE)
    returns = compute_returns(prices, "log")
    RESULTS["data"] = {
        "n_assets": int(returns.shape[1]),
        "n_days": int(returns.shape[0]),
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
    }
    log(f"  returns {returns.shape}, {returns.index[0].date()} – {returns.index[-1].date()}")
    return returns


def stage_spectrum(returns):
    """Full-sample spectrum + rolling MP counts + spectrum-wide reversion test."""
    log("Stage 2: spectrum-wide reversion test")
    T, N = returns.shape
    full = decompose(returns)
    w = returns.iloc[-BASE["window"] :]
    roll = decompose(w)
    ew = np.ones(N) / np.sqrt(N)
    RESULTS["spectrum"] = {
        "full_q": full.q, "full_mp_upper": full.mp_upper, "full_n_signal": full.n_signal_modes,
        "full_lambda0": float(full.eigenvalues[0]),
        "full_lambda0_share": float(full.eigenvalues[0] / N),
        "full_mode0_ew_overlap": float(abs(full.eigenvectors[:, 0] @ ew)),
        "roll_q": roll.q, "roll_mp_upper": roll.mp_upper,
    }

    # Figure: eigenvalue density vs MP (full sample) and rank plot (rolling).
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    lam = full.eigenvalues
    q = full.q
    # Laloux et al. (1999): the bulk only carries the variance not
    # absorbed by the market mode, so rescale sigma^2 = 1 - lambda_0 / N.
    s2 = 1 - lam[0] / N
    lm, lp = marchenko_pastur_bounds(T, N, s2)
    RESULTS["spectrum"]["full_mp_upper_rescaled"] = float(lp)
    RESULTS["spectrum"]["full_n_signal_rescaled"] = int((lam > lp).sum())
    ax.hist(lam[lam < 6], bins=80, density=True, color=C["blue"], alpha=0.55, label="Empirical")
    x = np.linspace(lm, lp, 400)
    rho = np.sqrt(np.maximum((lp - x) * (x - lm), 0)) / (2 * np.pi * q * s2 * x)
    ax.plot(x, rho, color=C["red"], lw=1.2, label=f"MP, $\\sigma^2$={s2:.2f}")
    ax.set_xlabel(r"Eigenvalue $\lambda$")
    ax.set_ylabel("Density")
    ax.set_title(f"Full sample (T={T}, N={N}); $\\lambda_0$={lam[0]:.0f} (off scale)", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    lr = roll.eigenvalues[: min(BASE['window'] - 1, N)]
    ax.semilogy(np.arange(len(lr)), lr, ".", ms=2.5, color=C["blue"])
    ax.axhline(roll.mp_upper, color=C["red"], ls="--", lw=1, label=r"$\lambda_+$")
    ax.set_xlabel("Mode rank $k$")
    ax.set_ylabel(r"$\lambda_k$ (log)")
    ax.set_title(f"Last 252-day window (q={roll.q:.2f}): {roll.n_signal_modes} above $\\lambda_+$",
                 fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "spectrum")

    out = {}
    for sig in ("level", "return"):
        _, panel_all, spec = build_episodes(
            returns, n_candidates=None, collect_all_modes=True, signal=sig, **BASE
        )
        out[sig] = (panel_all, spec)

    # Common OOS period for the test: same as strategies (after MIN_TRAIN).
    panel_all, spec = out["level"]
    n_sig = spec["n_signal"]
    keep = spec["episode_of_day"] >= MIN_TRAIN
    rank = spec["pnl"].shape[1]

    def sr(x):
        return x.mean() / x.std(ddof=1) * np.sqrt(252)

    # Buckets defined relative to each episode's own MP edge.
    buckets = [
        ("Market (k=0)", lambda e: [0]),
        ("Signal, above MP (1..n_s-1)", lambda e: list(range(1, n_sig[e]))),
        ("Edge bulk (n_s..n_s+24)", lambda e: list(range(n_sig[e], n_sig[e] + 25))),
        ("Upper bulk (n_s+25..99)", lambda e: list(range(n_sig[e] + 25, 100))),
        ("Middle bulk (100..174)", lambda e: list(range(100, 175))),
        ("Lower bulk (175..rank)", lambda e: list(range(175, rank))),
    ]
    rows = []
    for sig in ("level", "return"):
        pn = out[sig][1]["pnl"]
        rt = out[sig][1]["ret"]
        epd = out[sig][1]["episode_of_day"]
        for name, f in buckets:
            daily = np.full(len(pn), np.nan)
            for e in np.unique(epd):
                ks = f(e)
                rows_e = epd == e
                daily[rows_e] = pn[rows_e][:, ks].mean(axis=1)
            d = daily[keep]
            rows.append({"signal": sig, "bucket": name, "sharpe": sr(d),
                         "t_nw": newey_west_tstat(d, 10), "mean_bp": d.mean() * 1e4})
        if sig == "level":
            # OOS lag-1 autocorrelation of mode returns by bucket (pooled).
            for i, (name, f) in enumerate(buckets):
                acs = []
                for e in np.unique(epd):
                    if e < MIN_TRAIN:
                        continue
                    ks = f(e)
                    x = rt[epd == e][:, ks]
                    xc = x - x.mean(axis=0)
                    acs.append(((xc[1:] * xc[:-1]).sum(0) / (xc**2).sum(0)).mean())
                rows[i]["oos_ac1"] = float(np.mean(acs))
                rows[i]["oos_ac1_t"] = float(np.mean(acs) / (np.std(acs, ddof=1) / np.sqrt(len(acs))))
    bt = pd.DataFrame(rows)
    bt.to_csv(OUT / "spectrum_buckets.csv", index=False)
    RESULTS["spectrum_buckets"] = bt.to_dict(orient="records")

    # Table.
    lv = bt[bt.signal == "level"].set_index("bucket")
    rv = bt[bt.signal == "return"].set_index("bucket")
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Level reversion} & \multicolumn{2}{c}{Return reversal} & \multicolumn{2}{c}{OOS lag-1 AC} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Modes & Sharpe & $t_{NW}$ & Sharpe & $t_{NW}$ & $\bar\rho_1$ & $t$ \\", r"\midrule"]
    for name, _ in buckets:
        a, b = lv.loc[name], rv.loc[name]
        label = TEX_BUCKET.get(name, name)
        lines.append(f"{label} & {fmt(a.sharpe)} & {fmt(a.t_nw)} & {fmt(b.sharpe)} & {fmt(b.t_nw)} & "
                     f"{fmt(a.oos_ac1, 3)} & {fmt(a.oos_ac1_t)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "spectrum_buckets.tex").write_text("\n".join(lines))

    # Figure: OOS Sharpe by mode rank (smoothed), both signals.
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for sig, col, lab in (("level", C["blue"], "Level reversion (this paper)"),
                          ("return", C["orange"], "Daily-return reversal (original spec.)")):
        pn = out[sig][1]["pnl"][keep]
        s = pn.mean(0) / pn.std(0, ddof=1) * np.sqrt(252)
        sm = pd.Series(s).rolling(9, center=True, min_periods=1).mean()
        ax.plot(s, ".", ms=2, color=col, alpha=0.35)
        ax.plot(sm, color=col, lw=1.4, label=lab)
    ax.axvline(np.mean(n_sig[MIN_TRAIN:]), color=C["red"], ls="--", lw=1,
               label=r"Mean MP edge $\bar n_s$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Mode rank $k$ (descending eigenvalue)")
    ax.set_ylabel("OOS Sharpe, standalone")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "spectrum_reversion")

    # Theory check: OU relaxation. In-window half-life vs eigenvalue.
    pa = panel_all
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    g = pa.assign(lb=pd.qcut(pa["log_lambda"], 30)).groupby("lb", observed=True)
    xm, ym = g["log_lambda"].median(), g["log_half_life"].median()
    ax.plot(xm, ym, "o", ms=3, color=C["blue"])
    cap = np.log(252)
    sub = pa[pa.log_half_life < cap - 1e-6]
    slope, icpt = np.polyfit(sub["log_lambda"], sub["log_half_life"], 1)
    xx = np.linspace(xm.min(), xm.max(), 10)
    ax.plot(xx, icpt + slope * xx, color=C["red"], lw=1, label=f"OLS slope {slope:.2f}")
    ax.set_xlabel(r"$\log \lambda_k$")
    ax.set_ylabel(r"$\log$ half-life (days)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "ou_halflife")
    frac_capped = float((pa.log_half_life >= cap - 1e-6).mean())
    rho = pa[["log_lambda", "log_half_life"]].corr(method="spearman").iloc[0, 1]
    RESULTS["ou_check"] = {"slope": float(slope), "spearman": float(rho), "frac_no_reversion": frac_capped}
    log(f"  OU check: slope {slope:.3f}, spearman {rho:.3f}")
    return out


def stage_regimes(returns):
    log("Stage 3: regimes")
    mfeat = market_features(returns)
    probs, fits = walk_forward_regimes(mfeat, first_fit="2011-01-01")
    last = fits[-1]
    RESULTS["regime"] = {
        "first_year": fits[0]["year"],
        "calm_vol": float(np.exp(last["means"][0, 0])),
        "stress_vol": float(np.exp(last["means"][1, 0])),
        "calm_corr": float(last["means"][0, 1]),
        "stress_corr": float(last["means"][1, 1]),
        "p_stay_calm": float(last["trans"][0, 0]),
        "p_stay_stress": float(last["trans"][1, 1]),
        "frac_stress": float((probs.p_stress > 0.5).mean()),
        "stress_by_year": {int(k.year): float(v) for k, v in (probs.p_stress > 0.5).resample("YE").mean().items()},
    }
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1.3, 1]})
    a1.plot(np.exp(mfeat.log_vol), color=C["grey"], lw=0.7, label="21d market vol (ann.)")
    a1b = a1.twinx()
    a1b.plot(mfeat.avg_corr, color=C["blue"], lw=0.7, label="63d avg. correlation")
    a1.set_ylabel("Vol")
    a1b.set_ylabel("Avg corr", color=C["blue"])
    a1b.grid(False)
    a1.legend(frameon=False, fontsize=7, loc="upper left")
    a2.fill_between(probs.index, probs.p_stress, color=C["red"], alpha=0.5, lw=0)
    a2.set_ylabel(r"$P(\mathrm{stress})$")
    a2.set_ylim(0, 1)
    fig.tight_layout()
    save(fig, "regimes")
    return mfeat, probs


def stage_prediction(returns, probs, mfeat, pools):
    log("Stage 4: mode-quality prediction")
    data = {}
    auc_rows = []
    for pool, ncand in pools.items():
        eps, panel, _ = build_episodes(returns, n_candidates=ncand, **BASE)
        panel = attach_regimes(panel, probs, mfeat)
        for model in ("logit", "gbm", "rf"):
            for fs_name, fs in (("mode", FEATURES_MODE), ("all", FEATURES_ALL)):
                t0 = time.time()
                re = 1 if model == "logit" else 3
                col = f"p_{model}_{fs_name}"
                panel[col] = walk_forward_predict(panel, fs, model, MIN_TRAIN, retrain_every=re)
                auc = oos_auc(panel, col)
                lo, hi = block_bootstrap_auc(panel, col, n_boot=500)
                auc_rows.append({"pool": pool, "model": model, "features": fs_name,
                                 "auc": auc, "lo": lo, "hi": hi})
                log(f"  {pool:5s} {model:5s} {fs_name:4s} AUC {auc:.3f} [{lo:.3f},{hi:.3f}] "
                    f"({time.time() - t0:.0f}s)")
        # Heuristic score: short in-window half-life.
        panel["h_halflife"] = -panel["log_half_life"]
        panel["h_lambda"] = -panel["log_lambda"]
        for h in ("h_halflife", "h_lambda"):
            m = panel.episode >= MIN_TRAIN
            auc_rows.append({"pool": pool, "model": h, "features": "-", "auc": oos_auc(panel[m], h),
                             "lo": np.nan, "hi": np.nan})
        data[pool] = (eps, panel)
        oos = panel[panel.episode >= MIN_TRAIN]
        RESULTS.setdefault("panel", {})[pool] = {
            "rows": int(len(panel)), "oos_rows": int(len(oos)),
            "episodes": int(panel.episode.nunique()),
            "base_rate": float(oos.y.mean()),
            "frac_above_mp": float(panel.above_mp.mean()),
            "first_oos_date": str(oos.date.min().date()),
        }
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(OUT / "auc.csv", index=False)
    RESULTS["auc"] = auc.to_dict(orient="records")

    # Feature-group ablation (primary model, all features).
    abl = []
    for pool, (eps, panel) in data.items():
        base = oos_auc(panel, f"p_{PRIMARY_MODEL}_all")
        for g, cols in GROUPS.items():
            fs = [f for f in FEATURES_ALL if f not in cols]
            p = walk_forward_predict(panel, fs, PRIMARY_MODEL, MIN_TRAIN)
            tmp = panel.assign(_p=p)
            abl.append({"pool": pool, "group": g, "delta_auc": oos_auc(tmp, "_p") - base})
            # Single-group model.
            p1 = walk_forward_predict(panel, cols, PRIMARY_MODEL, MIN_TRAIN)
            abl[-1]["solo_auc"] = oos_auc(panel.assign(_p=p1), "_p")
    abl = pd.DataFrame(abl)
    abl.to_csv(OUT / "ablation.csv", index=False)
    RESULTS["ablation"] = abl.to_dict(orient="records")

    # Logistic coefficients on the full sample (descriptive only).
    from src.selection import make_model
    coefs = {}
    for pool, (eps, panel) in data.items():
        mdl = make_model("logit").fit(panel[FEATURES_ALL], panel["y"])
        coefs[pool] = dict(zip(FEATURES_ALL, mdl[-1].coef_[0].round(4).tolist()))
    RESULTS["logit_coefs"] = coefs

    # AUC table.
    lines = [r"\begin{tabular}{llcc}", r"\toprule",
             r"Pool & Model & Mode features & Mode + regime features \\", r"\midrule"]
    names = {"logit": "Logistic (L2)", "gbm": "Gradient boosting", "rf": "Random forest",
             "h_halflife": "Heuristic: $-\\log$ half-life", "h_lambda": "Heuristic: $-\\log\\lambda$"}
    pool_names = {"A": "Modes 1--30", "B": "Full spectrum"}
    for pool in data:
        first = True
        for model in ("logit", "gbm", "rf", "h_halflife", "h_lambda"):
            r = auc[(auc.pool == pool) & (auc.model == model)]
            pn = pool_names[pool] if first else ""
            first = False
            if model.startswith("h_"):
                a = r.iloc[0]
                lines.append(f"{pn} & {names[model]} & \\multicolumn{{2}}{{c}}{{{a.auc:.3f}}} \\\\")
            else:
                cells = []
                for fs in ("mode", "all"):
                    a = r[r.features == fs].iloc[0]
                    cells.append(f"{a.auc:.3f} [{a.lo:.3f}, {a.hi:.3f}]")
                lines.append(f"{pn} & {names[model]} & {cells[0]} & {cells[1]} \\\\")
        lines.append(r"\midrule" if pool != list(data)[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (TAB / "auc.tex").write_text("\n".join(lines))

    # Ablation table.
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Modes 1--30} & \multicolumn{2}{c}{Full spectrum} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Feature group & Solo AUC & $\Delta$AUC (drop) & Solo AUC & $\Delta$AUC (drop) \\", r"\midrule"]
    for g in GROUPS:
        a = abl[(abl.pool == "A") & (abl.group == g)].iloc[0]
        b = abl[(abl.pool == "B") & (abl.group == g)].iloc[0]
        lines.append(f"{g} & {a.solo_auc:.3f} & {a.delta_auc:+.3f} & {b.solo_auc:.3f} & {b.delta_auc:+.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "ablation.tex").write_text("\n".join(lines))

    # AUC by year figure (primary model, both pools).
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    from sklearn.metrics import roc_auc_score
    for (pool, (eps, panel)), col in zip(data.items(), (C["blue"], C["green"])):
        oos = panel[panel.episode >= MIN_TRAIN].copy()
        yr = oos.groupby(oos.date.dt.year).apply(
            lambda g: roc_auc_score(g.y, g[f"p_{PRIMARY_MODEL}_all"]) if g.y.nunique() == 2 else np.nan)
        ax.plot(yr.index, yr.values, "o-", ms=3, color=col, label=pool_names[pool])
    ax.axhline(0.5, color="k", lw=0.6)
    ax.set_ylabel("OOS AUC")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "auc_by_year")

    # AUC / base rate by regime.
    reg_rows = []
    for pool, (eps, panel) in data.items():
        oos = panel[panel.episode >= MIN_TRAIN]
        for st, g in oos.groupby(oos.p_stress > 0.5):
            reg_rows.append({"pool": pool, "stress": bool(st), "n_episodes": int(g.episode.nunique()),
                             "base_rate": float(g.y.mean()),
                             "mean_fwd_pnl_norm": float(g.fwd_pnl_norm.mean()),
                             "auc": oos_auc(g, f"p_{PRIMARY_MODEL}_all")})
    RESULTS["by_regime"] = reg_rows
    return data


def stage_strategies(returns, data, probs):
    log("Stage 5: strategies")
    strat = {}
    common_start = None
    for pool, (eps, panel) in data.items():
        start = panel.loc[panel.episode == MIN_TRAIN, "date"].iloc[0]
        common_start = start
        oos = panel.episode >= MIN_TRAIN
        ones = pd.Series(1.0, index=panel.index).where(oos, 0.0)
        specs = {}
        if pool == "A":
            specs["MP signal modes, equal weight"] = ones * panel["above_mp"]
        specs["All candidates, equal weight"] = ones
        specs["Heuristic: shortest half-life"] = select_top_fraction(panel, "h_halflife", FRAC, oos)
        specs["ML: mode features"] = select_top_fraction(panel, f"p_{PRIMARY_MODEL}_mode", FRAC, oos)
        specs["ML: mode + regime features"] = select_top_fraction(panel, f"p_{PRIMARY_MODEL}_all", FRAC, oos)
        res = {k: run_strategy(eps, panel, w, returns, start) for k, w in specs.items()}
        # Regime overlay on ML + regime.
        ml = res["ML: mode + regime features"]
        expo = regime_exposure(ml["pnl_gross"].reindex(returns.index).fillna(0.0), panel, eps)
        res["ML + regime + exposure overlay"] = run_strategy(
            eps, panel, specs["ML: mode + regime features"], returns, start, exposure=expo)
        eqr = res["All candidates, equal weight"]
        expo_eq = regime_exposure(eqr["pnl_gross"].reindex(returns.index).fillna(0.0), panel, eps)
        res["Equal weight + exposure overlay"] = run_strategy(
            eps, panel, specs["All candidates, equal weight"], returns, start, exposure=expo_eq)
        RESULTS.setdefault("overlay_eq", {})[pool] = {
            "mean_exposure": float(expo_eq[expo_eq.index >= MIN_TRAIN].mean())}
        pd.DataFrame({"ml": expo, "eq": expo_eq}).to_csv(OUT / f"exposure_{pool}.csv")
        RESULTS.setdefault("overlay", {})[pool] = {
            "mean_exposure": float(expo[expo.index >= MIN_TRAIN].mean()),
            "min_exposure": float(expo[expo.index >= MIN_TRAIN].min())}
        strat[pool] = (res, specs)

        # Placebo: random selection of the same number of modes.
        rng = np.random.default_rng(1)
        pl = []
        for b in range(200):
            rnd = pd.Series(rng.random(len(panel)), index=panel.index)
            panel["_rnd"] = rnd
            w = select_top_fraction(panel, "_rnd", FRAC, oos)
            pl.append(run_strategy(eps, panel, w, returns, start)["metrics"]["sharpe"])
        panel.drop(columns="_rnd", inplace=True)
        pl = np.array(pl)
        for k in ("ML: mode features", "ML: mode + regime features"):
            s = res[k]["metrics"]["sharpe"]
            RESULTS.setdefault("placebo", {}).setdefault(pool, {})[k] = {
                "sharpe": s, "placebo_mean": float(pl.mean()), "placebo_p95": float(np.percentile(pl, 95)),
                "p_value": float((pl >= s).mean())}
        log(f"  pool {pool}: placebo mean {pl.mean():.2f}, 95th {np.percentile(pl, 95):.2f}")

        # Bootstrap Sharpe differences vs equal weight.
        for k in ("Heuristic: shortest half-life", "ML: mode features", "ML: mode + regime features",
                  "ML + regime + exposure overlay", "Equal weight + exposure overlay"):
            d, lo, p = sharpe_diff_bootstrap(res[k]["pnl"], res["All candidates, equal weight"]["pnl"])
            RESULTS.setdefault("sharpe_diff", {}).setdefault(pool, {})[k] = {"diff": d, "lo": lo, "p": p}

    # Original repo baseline on the same data and period.
    log("  original baseline")
    orig = walk_forward_backtest(returns, estimation_window=252, reestimation_freq=21,
                                 signal_lookback=20, market_mode_skip=1, max_modes=20,
                                 transaction_cost_bps=COST_BPS)
    orig_res = evaluate(orig.positions.shift(1).fillna(0.0), returns, COST_BPS, common_start)

    # Strategy table.
    metrics_rows = [("Original repo specification", "--", orig_res["metrics"])]
    for pool, (res, _) in strat.items():
        for k, r in res.items():
            metrics_rows.append((k, pool, r["metrics"]))
    for _, _, m in metrics_rows:
        m["breakeven_bps"] = m["gross_ann_return"] / (252 * m["avg_turnover"]) * 1e4
    RESULTS["strategies"] = [{"name": n, "pool": p, **m} for n, p, m in metrics_rows]
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Strategy & Pool & Ann. ret. & Ann. vol & Sharpe & Gross SR & Max DD & Turnover & Break-even \\",
             r"\midrule"]
    pn = {"A": "1--30", "B": "Full", "--": "--"}
    last_pool = None
    for n, p, m in metrics_rows:
        if last_pool is not None and p != last_pool:
            lines.append(r"\midrule")
        last_pool = p
        lines.append(f"{n} & {pn[p]} & {100 * m['annualized_return']:.1f}\\% & {100 * m['annualized_vol']:.1f}\\% & "
                     f"{m['sharpe']:.2f} & {m['gross_sharpe']:.2f} & {100 * m['max_drawdown']:.1f}\\% & "
                     f"{m['avg_turnover']:.2f} & {m['breakeven_bps']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "strategies.tex").write_text("\n".join(lines))

    # Cumulative PnL figure: gross (top) and net of 2bp (bottom).
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    cols = {"All candidates, equal weight": C["grey"], "Heuristic: shortest half-life": C["orange"],
            "ML: mode features": C["blue"], "ML: mode + regime features": C["green"],
            "ML + regime + exposure overlay": C["purple"], "MP signal modes, equal weight": C["red"],
            "Equal weight + exposure overlay": "#0891b2"}
    for c, (pool, (res, _)) in enumerate(strat.items()):
        for r_, kind in enumerate(("pnl_gross", "pnl")):
            ax = axes[r_, c]
            for k, r in res.items():
                ax.plot(r[kind].cumsum() * 100, lw=0.9, color=cols[k], label=k)
            ax.axhline(0, color="k", lw=0.5)
            if r_ == 0:
                ax.set_title({"A": "Pool A: modes 1–30", "B": "Pool B: full spectrum"}[pool], fontsize=9)
            if c == 0:
                ax.set_ylabel(("Gross" if kind == "pnl_gross" else f"Net of {COST_BPS:.0f}bp")
                              + " cum. return (%)")
    h, l = [], []
    for ax in axes.ravel():
        for a_, b_ in zip(*ax.get_legend_handles_labels()):
            if b_ not in l:
                h.append(a_); l.append(b_)
    fig.legend(h, l, frameon=False, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, "cumulative_pnl")

    # Exposure overlay figure.
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    for pool, col in (("A", C["blue"]), ("B", C["green"])):
        e = pd.read_csv(OUT / f"exposure_{pool}.csv", index_col=0)["ml"]
        dates = data[pool][1].groupby("episode")["date"].first()
        ax.step(dates.reindex(e.index), e.values, where="post", color=col, lw=1,
                label={"A": "Pool A", "B": "Pool B"}[pool])
    ax.set_ylabel("Exposure $e_j$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "exposure")

    # Performance by regime.
    reg_perf = []
    pst = probs["p_stress"].reindex(returns.index).shift(1) > 0.5
    for pool, (res, _) in strat.items():
        for k in ("All candidates, equal weight", "Equal weight + exposure overlay",
                  "ML: mode + regime features", "ML + regime + exposure overlay"):
            pnl = res[k]["pnl"]
            for st in (False, True):
                x = pnl[pst.reindex(pnl.index).fillna(False) == st]
                reg_perf.append({"pool": pool, "strategy": k, "stress": st, "days": int(len(x)),
                                 "sharpe": float(x.mean() / x.std() * np.sqrt(252)),
                                 "ann_ret": float(x.mean() * 252)})
    RESULTS["regime_perf"] = reg_perf
    return strat, orig_res, common_start


def stage_robustness(returns, probs, mfeat, data, strat, common_start):
    log("Stage 6: robustness")
    rows = []
    key = "ML: mode + regime features"
    eq = "All candidates, equal weight"

    def add(name, pool, ml, eqm):
        rows.append({"test": name, "pool": pool, "ml_sharpe": ml["sharpe"], "eq_sharpe": eqm["sharpe"],
                     "ml_gross": ml["gross_sharpe"], "eq_gross": eqm["gross_sharpe"],
                     "ml_turnover": ml["avg_turnover"], "ml_maxdd": ml["max_drawdown"]})

    # (a) Costs.
    cost_rows = []
    for pool, (res, specs) in strat.items():
        eps, panel = data[pool]
        for c in (0, 1, 2, 5, 10):
            for k in (eq, "Heuristic: shortest half-life", key):
                r = evaluate(res[k]["positions"], returns, c, common_start)["metrics"]
                cost_rows.append({"pool": pool, "strategy": k, "cost_bps": c, "sharpe": r["sharpe"]})
    cost = pd.DataFrame(cost_rows)
    RESULTS["costs"] = cost.to_dict(orient="records")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    for ax, pool in zip(axes, ("A", "B")):
        for k, col in ((eq, C["grey"]), ("Heuristic: shortest half-life", C["orange"]), (key, C["green"])):
            s = cost[(cost.pool == pool) & (cost.strategy == k)]
            ax.plot(s.cost_bps, s.sharpe, "o-", ms=3, color=col, label=k)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("Cost (bp per unit turnover)")
        ax.set_title({"A": "Pool A: modes 1–30", "B": "Pool B: full spectrum"}[pool], fontsize=9)
    axes[0].set_ylabel("Net Sharpe")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "costs")

    # (b) Sub-periods.
    for pool, (res, _) in strat.items():
        for a, b in (("2013", "2016"), ("2017", "2019"), ("2020", "2021"), ("2022", "2024")):
            m = {}
            for k in (key, eq):
                x = res[k]["pnl"][a:b]
                g = res[k]["pnl_gross"][a:b]
                m[k] = {"sharpe": x.mean() / x.std() * np.sqrt(252), "avg_turnover": np.nan,
                        "gross_sharpe": g.mean() / g.std() * np.sqrt(252),
                        "max_drawdown": float((x.cumsum() - x.cumsum().cummax()).min())}
            add(f"Sub-period {a}--{b}", pool, m[key], m[eq])

    # (c) Selection fraction and model.
    for pool, (eps, panel) in data.items():
        oos = panel.episode >= MIN_TRAIN
        eqm = strat[pool][0][eq]["metrics"]
        for f in (0.2, 0.5):
            w = select_top_fraction(panel, f"p_{PRIMARY_MODEL}_all", f, oos)
            add(f"Select top {int(f * 100)}\\%", pool, run_strategy(eps, panel, w, returns, common_start)["metrics"], eqm)
        for model in ("gbm", "rf"):
            w = select_top_fraction(panel, f"p_{model}_all", FRAC, oos)
            add(f"Model: {model.upper()}", pool, run_strategy(eps, panel, w, returns, common_start)["metrics"], eqm)

    # (d) Re-built episodes: delay, window, lookback, holding period.
    variants = {
        "Execution delay +1 day": dict(delay=1),
        "Window $W=126$": dict(window=126),
        "Window $W=504$": dict(window=504),
        "Lookback $L=20$": dict(lookback=20),
        "Lookback $L=120$": dict(lookback=120),
        "Holding $H=10$": dict(hold=10),
        "Holding $H=42$": dict(hold=42),
    }
    pools = {"A": 30, "B": None}
    for name, kw in variants.items():
        cfg = {**BASE, **kw}
        if cfg["lookback"] + 63 > cfg["window"]:
            cfg["lookback"] = min(cfg["lookback"], cfg["window"] - 63)
        for pool, nc in pools.items():
            t0 = time.time()
            eps, panel, _ = build_episodes(returns, n_candidates=nc, **cfg)
            panel = attach_regimes(panel, probs, mfeat)
            min_train = int(round(MIN_TRAIN * BASE["hold"] / cfg["hold"]))
            panel["p"] = walk_forward_predict(panel, FEATURES_ALL, PRIMARY_MODEL, min_train)
            oos = panel.episode >= min_train
            # Evaluate on the common period so numbers are comparable.
            st = max(pd.Timestamp(common_start), panel.loc[oos, "date"].min())
            w = select_top_fraction(panel, "p", FRAC, oos)
            ml = run_strategy(eps, panel, w, returns, st)["metrics"]
            eqm = run_strategy(eps, panel, pd.Series(1.0, index=panel.index).where(oos, 0.0), returns, st)["metrics"]
            add(name, pool, ml, eqm)
            rows[-1]["auc"] = oos_auc(panel[oos], "p")
            log(f"  {name:28s} pool {pool}: ML SR {ml['sharpe']:.2f} vs EQ {eqm['sharpe']:.2f} "
                f"AUC {rows[-1]['auc']:.3f} ({time.time() - t0:.0f}s)")
    rob = pd.DataFrame(rows)
    rob.to_csv(OUT / "robustness.csv", index=False)
    RESULTS["robustness"] = rob.to_dict(orient="records")

    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{Pool A: modes 1--30} & \multicolumn{3}{c}{Pool B: full spectrum} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Test & ML & EW & AUC & ML & EW & AUC \\", r"\midrule"]
    base_a = strat["A"][0]
    base_b = strat["B"][0]
    def pair(n, g):
        return f"{n:.2f} ({g:.2f})"

    ma, ea = base_a[key]["metrics"], base_a[eq]["metrics"]
    mb, eb = base_b[key]["metrics"], base_b[eq]["metrics"]
    lines.append(f"Baseline ($W$=252, $L$=60, $H$=21) & {pair(ma['sharpe'], ma['gross_sharpe'])} & "
                 f"{pair(ea['sharpe'], ea['gross_sharpe'])} & {oos_auc(data['A'][1], 'p_logit_all'):.3f} & "
                 f"{pair(mb['sharpe'], mb['gross_sharpe'])} & {pair(eb['sharpe'], eb['gross_sharpe'])} & "
                 f"{oos_auc(data['B'][1], 'p_logit_all'):.3f} \\\\")
    lines.append(r"\midrule")
    for t in rob.test.unique():
        a = rob[(rob.test == t) & (rob.pool == "A")].iloc[0]
        b = rob[(rob.test == t) & (rob.pool == "B")].iloc[0]
        lines.append(f"{t} & {pair(a.ml_sharpe, a.ml_gross)} & {pair(a.eq_sharpe, a.eq_gross)} & "
                     f"{fmt(a.get('auc', np.nan), 3)} & {pair(b.ml_sharpe, b.ml_gross)} & "
                     f"{pair(b.eq_sharpe, b.eq_gross)} & {fmt(b.get('auc', np.nan), 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "robustness.tex").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    for d in (FIG, TAB, OUT):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    returns = stage_data()
    stage_spectrum(returns)
    mfeat, probs = stage_regimes(returns)
    data = stage_prediction(returns, probs, mfeat, {"A": 30, "B": None})
    strat, orig_res, common_start = stage_strategies(returns, data, probs)
    RESULTS["oos_start"] = str(pd.Timestamp(common_start).date())
    if not args.quick:
        stage_robustness(returns, probs, mfeat, data, strat, common_start)
    RESULTS["config"] = {**BASE, "min_train_episodes": MIN_TRAIN, "frac": FRAC, "cost_bps": COST_BPS,
                         "primary_model": PRIMARY_MODEL}

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    (OUT / "results.json").write_text(json.dumps(clean(RESULTS), indent=2))
    log(f"Done in {time.time() - t0:.0f}s. Results in {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
