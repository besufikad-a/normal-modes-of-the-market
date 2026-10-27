"""Prepare the (small) datasets used by the Manim scenes.

Outputs to ``animations/data/``:

* ``history.csv`` — monthly 1975–present: growth of $1 in the S&P 500
  (price index), a constant-maturity 10-year Treasury (total return
  reconstructed from yields), and gold; monthly returns; and rolling
  36-month correlations for each pair.
* ``market_modes.json`` — correlation matrix, eigenvalues and
  eigenvectors of 12 large S&P 500 stocks (daily returns 2015–2024),
  plus cumulative mode "prices" for the market mode and a low-variance mode.
* ``spectrum.json`` — rolling 252-day eigenvalue spectra of the 417-stock
  universe used in the paper (one per month), with the Marchenko–Pastur
  edge.

Sources: Yahoo Finance (^GSPC, ^TNX, daily, resampled to month-end),
datahub.io gold-prices (monthly), and the study's price cache.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HIST = ROOT / "data" / "cache" / "history"
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

START = "1975-01-31"


def bond_total_return(yield_pct: pd.Series) -> pd.Series:
    """Monthly total return of a constant-maturity 10y par bond.

    Each month we buy a 10-year par bond at last month's yield, earn one
    month of coupon, and reprice it at this month's yield (semiannual
    compounding, 20 periods; the one-month roll-down is ignored).
    """
    y0 = yield_pct.shift(1) / 100
    y1 = yield_pct / 100
    c = y0 / 2
    n = 20
    disc = (1 + y1 / 2) ** -n
    price = c / (y1 / 2) * (1 - disc) + disc
    return (price - 1 + y0 / 12).dropna()


def load_history() -> pd.DataFrame:
    spx = pd.read_csv(HIST / "GSPC_monthly.csv", index_col=0, parse_dates=True).squeeze()
    tnx = pd.read_csv(HIST / "TNX_monthly.csv", index_col=0, parse_dates=True).squeeze()
    gold = pd.read_csv(HIST / "gold_monthly.csv")
    gold.index = pd.to_datetime(gold["Date"]) + pd.offsets.MonthEnd(0)
    gold = gold["Price"]
    spx.index = spx.index + pd.offsets.MonthEnd(0)
    tnx.index = tnx.index + pd.offsets.MonthEnd(0)

    df = pd.DataFrame({"spx": spx, "tnx": tnx, "gold": gold}).dropna()
    df = df[df.index >= pd.Timestamp(START) - pd.offsets.MonthEnd(1)]
    rets = pd.DataFrame({
        "r_stock": df["spx"].pct_change(),
        "r_bond": bond_total_return(df["tnx"]),
        "r_gold": df["gold"].pct_change(),
    }).dropna()
    out = rets.copy()
    for a in ("stock", "bond", "gold"):
        out[f"v_{a}"] = (1 + rets[f"r_{a}"]).cumprod()
    out["yield10"] = df["tnx"].reindex(out.index)
    w = 36
    out["c_stock_bond"] = rets["r_stock"].rolling(w).corr(rets["r_bond"])
    out["c_stock_gold"] = rets["r_stock"].rolling(w).corr(rets["r_gold"])
    out["c_bond_gold"] = rets["r_bond"].rolling(w).corr(rets["r_gold"])
    out["year"] = out.index.year + (out.index.month - 0.5) / 12
    out.index.name = "date"
    return out


def market_modes() -> dict:
    prices = pd.read_parquet(ROOT / "data" / "cache" / "sp500_prices_2010_2024.parquet")
    names = ["AAPL", "MSFT", "NVDA", "JPM", "BAC", "GS", "XOM", "CVX", "JNJ", "PFE", "KO", "PG"]
    sectors = ["Tech", "Tech", "Tech", "Fin", "Fin", "Fin", "Energy", "Energy",
               "Health", "Health", "Staples", "Staples"]
    p = prices[names]["2015":"2024"]
    r = np.log(p / p.shift(1)).dropna()
    z = (r - r.mean()) / r.std()
    C = z.corr().values
    lam, V = np.linalg.eigh(C)
    lam, V = lam[::-1], V[:, ::-1]
    for k in range(V.shape[1]):
        if V[np.argmax(np.abs(V[:, k])), k] < 0:
            V[:, k] *= -1
    M = z.values @ V  # demeaned mode returns (drift removed for illustration)
    X = np.cumsum(M, axis=0)
    step = 5  # weekly samples for plotting
    day = z.abs().mean(axis=1).idxmax()  # most extreme day in the sample
    zd = z.loc[day].values
    return {
        "day": str(day.date()),
        "day_z": zd.round(4).tolist(),
        "day_coef": (V.T @ zd).round(4).tolist(),
        "names": names,
        "sectors": sectors,
        "corr": C.round(4).tolist(),
        "eigenvalues": lam.round(5).tolist(),
        "eigenvectors": V.round(5).tolist(),
        "mode_price_t": (np.arange(0, len(X), step) / 252 + 2015).round(4).tolist(),
        "mode_price": {str(k): (X[::step, k] / np.sqrt(lam[k])).round(4).tolist()
                       for k in (0, 1, 11)},
    }


def rolling_spectrum() -> dict:
    from src.data import compute_returns
    from src.modes import decompose

    prices = pd.read_parquet(ROOT / "data" / "cache" / "sp500_prices_2010_2024.parquet")
    r = compute_returns(prices)
    W, step, K = 252, 21, 60
    times, lams, nsig = [], [], []
    for t in range(W, len(r), step):
        m = decompose(r.iloc[t - W:t])
        times.append(float(r.index[t].year + r.index[t].dayofyear / 366))
        lams.append(m.eigenvalues[:K].round(4).tolist())
        nsig.append(int(m.n_signal_modes))
    return {"t": times, "lam": lams, "n_signal": nsig, "mp_upper": float(m.mp_upper),
            "N": int(r.shape[1]), "W": W}


def findings() -> dict:
    """Out-of-sample spectrum-bucket results from the paper (output/results.json)."""
    R = json.loads((ROOT / "output" / "results.json").read_text())
    b = [x for x in R["spectrum_buckets"] if x["signal"] == "level"]
    return {"buckets": [{"name": x["bucket"], "sharpe": x["sharpe"], "t": x["t_nw"],
                         "ac1": x.get("oos_ac1")} for x in b]}


if __name__ == "__main__":
    h = load_history()
    h.round(6).to_csv(OUT / "history.csv")
    print(f"history: {h.index[0].date()} – {h.index[-1].date()}, {len(h)} months")
    print(h[["v_stock", "v_bond", "v_gold"]].iloc[-1].round(2).to_dict())
    print("mean corr by decade:")
    print(h.groupby((h.index.year // 10) * 10)[["c_stock_bond", "c_stock_gold"]].mean().round(2))
    (OUT / "market_modes.json").write_text(json.dumps(market_modes()))
    (OUT / "findings.json").write_text(json.dumps(findings()))
    (OUT / "spectrum.json").write_text(json.dumps(rolling_spectrum()))
    print("wrote", sorted(p.name for p in OUT.iterdir()))
