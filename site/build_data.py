"""Bundle the data behind the interactive site into ``docs/data/``.

Inputs: ``animations/data/*`` (from animations/data_prep.py), the study's
price cache, and ``output/`` from scripts/run_study.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AD = ROOT / "animations" / "data"
OUT = ROOT / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def r(x, nd=5):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), nd) for v in x]


def history():
    h = pd.read_csv(AD / "history.csv", index_col=0, parse_dates=True)
    return {
        "date": [d.strftime("%Y-%m") for d in h.index],
        "year": r(h["year"], 4),
        "r_stock": r(h["r_stock"]), "r_bond": r(h["r_bond"]), "r_gold": r(h["r_gold"]),
        "yield10": r(h["yield10"], 3),
    }


def market():
    mm = json.loads((AD / "market_modes.json").read_text())
    prices = pd.read_parquet(ROOT / "data" / "cache" / "sp500_prices_2010_2024.parquet")
    p = prices[mm["names"]]["2015":"2024"]
    ret = np.log(p / p.shift(1)).dropna()
    z = (ret - ret.mean()) / ret.std()
    return {
        "names": mm["names"], "sectors": mm["sectors"], "corr": mm["corr"],
        "eigenvalues": mm["eigenvalues"], "eigenvectors": mm["eigenvectors"],
        "T": int(len(ret)),
        "dates": [d.strftime("%Y-%m-%d") for d in z.index],
        "z": [r(row, 3) for row in z.values],
    }


def strategies():
    d = pd.read_parquet(ROOT / "output" / "strategy_daily.parquet")
    d = d[d.index >= "2013-01-07"]
    out = {"dates": [x.strftime("%Y-%m-%d") for x in d.index], "p_stress": r(d["p_stress"], 3), "series": []}
    names = sorted({c.rsplit("|", 1)[0] for c in d.columns if "|" in c})
    for nm in names:
        pool, strat = nm.split("|", 1)
        out["series"].append({
            "pool": pool, "name": strat,
            "gross": r(d[f"{nm}|gross"].fillna(0), 7),
            "turnover": r(d[f"{nm}|turnover"].fillna(0), 4),
        })
    return out


def results():
    R = json.loads((ROOT / "output" / "results.json").read_text())
    keep = ["data", "spectrum_buckets", "auc", "ablation", "regime", "panel", "placebo", "sharpe_diff",
            "strategies", "ou_check", "by_regime", "costs", "robustness", "oos_start"]
    return {k: R[k] for k in keep if k in R}


if __name__ == "__main__":
    files = {
        "history.json": history(),
        "market.json": market(),
        "spectrum.json": json.loads((AD / "spectrum.json").read_text()),
        "strategies.json": strategies(),
        "results.json": results(),
    }
    for name, obj in files.items():
        (OUT / name).write_text(json.dumps(obj, separators=(",", ":")))
        print(f"{name}: {(OUT / name).stat().st_size / 1e3:.0f} kB")
