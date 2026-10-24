"""Write ``paper/numbers.tex``: LaTeX macros for every number quoted in the text.

Run after ``scripts/run_study.py``. Keeps the paper in sync with the code.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "output" / "results.json").read_text())

macros: dict[str, str] = {}


def m(name: str, value, nd: int = 2, pct: bool = False, sign: bool = False):
    if isinstance(value, str):
        macros[name] = value
        return
    v = 100 * value if pct else value
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    macros[name] = s.replace("-", "$-$") if not sign else s.replace("-", "$-$").replace("+", "$+$")


def strat(name, pool):
    for s in R["strategies"]:
        if s["name"] == name and s["pool"] == pool:
            return s
    raise KeyError((name, pool))


def auc(pool, model, feats):
    for a in R["auc"]:
        if a["pool"] == pool and a["model"] == model and a["features"] == feats:
            return a
    raise KeyError((pool, model, feats))


d = R["data"]
m("DataAssets", str(d["n_assets"]))
m("DataDays", f"{d['n_days']:,}".replace(",", "{,}"))
m("DataStart", "January 2010")
m("DataEnd", "December 2024")
m("NEpisodes", str(R["panel"]["A"]["episodes"]))
m("MinTrain", str(R["config"]["min_train_episodes"]))
m("OOSStart", "7 January 2013")
sp = R["spectrum"]
m("FullOverlap", sp["full_mode0_ew_overlap"], 3)
m("FullLamZero", sp["full_lambda0"], 0)
m("FullLamZeroShare", sp["full_lambda0_share"], 0, pct=True)
m("RollMPUpper", sp["roll_mp_upper"], 2)
m("RollQ", sp["roll_q"], 2)
m("FullNSigRescaled", str(sp["full_n_signal_rescaled"]))

b = {(x["signal"], x["bucket"]): x for x in R["spectrum_buckets"]}
names = {
    "Market": "Market (k=0)",
    "Signal": "Signal, above MP (1..n_s-1)",
    "Edge": "Edge bulk (n_s..n_s+24)",
    "Upper": "Upper bulk (n_s+25..99)",
    "Middle": "Middle bulk (100..174)",
    "Lower": "Lower bulk (175..rank)",
}
for short, full in names.items():
    m(f"SR{short}Level", b[("level", full)]["sharpe"])
    m(f"T{short}Level", b[("level", full)]["t_nw"])
    m(f"SR{short}Ret", b[("return", full)]["sharpe"])
    m(f"T{short}Ret", b[("return", full)]["t_nw"])
    m(f"AC{short}", b[("level", full)]["oos_ac1"], 3)
    m(f"ACt{short}", b[("level", full)]["oos_ac1_t"], 1)

ou = R["ou_check"]
m("OUSlope", ou["slope"])
m("OUSpearman", ou["spearman"])
m("OUNoRev", ou["frac_no_reversion"], 0, pct=True)

rg = R["regime"]
m("CalmVol", rg["calm_vol"], 0, pct=True)
m("StressVol", rg["stress_vol"], 0, pct=True)
m("CalmCorr", rg["calm_corr"])
m("StressCorr", rg["stress_corr"])
m("PStayCalm", rg["p_stay_calm"], 3)
m("PStayStress", rg["p_stay_stress"], 3)
m("FracStress", rg["frac_stress"], 0, pct=True)
m("StressTwentyTwenty", rg["stress_by_year"]["2020"], 0, pct=True)
m("StressTwentyTwentyTwo", rg["stress_by_year"]["2022"], 0, pct=True)
m("StressTwentySeventeen", rg["stress_by_year"]["2017"], 0, pct=True)

for pool in ("A", "B"):
    P = R["panel"][pool]
    m(f"BaseRate{pool}", P["base_rate"], 0, pct=True)
    m(f"OOSRows{pool}", f"{P['oos_rows']:,}".replace(",", "{,}"))
    for model in ("logit", "gbm", "rf"):
        for fs in ("mode", "all"):
            a = auc(pool, model, fs)
            tag = f"{model.capitalize()}{'Mode' if fs == 'mode' else 'All'}{pool}"
            m(f"AUC{tag}", a["auc"], 3)
            m(f"AUCLo{tag}", a["lo"], 3)
            m(f"AUCHi{tag}", a["hi"], 3)
    for h, nm in (("h_halflife", "HalfLife"), ("h_lambda", "Lambda")):
        m(f"AUCHeur{nm}{pool}", auc(pool, h, "-")["auc"], 3)

for r in R["by_regime"]:
    tag = f"{r['pool']}{'Stress' if r['stress'] else 'Calm'}"
    m(f"BaseRate{tag}", r["base_rate"], 0, pct=True)
    m(f"AUCReg{tag}", r["auc"], 3)
    m(f"NEp{tag}", str(r["n_episodes"]))

S = {
    "EW": "All candidates, equal weight",
    "MP": "MP signal modes, equal weight",
    "Heur": "Heuristic: shortest half-life",
    "MLMode": "ML: mode features",
    "MLAll": "ML: mode + regime features",
    "MLOver": "ML + regime + exposure overlay",
    "EWOver": "Equal weight + exposure overlay",
}
for pool in ("A", "B"):
    for tag, nm in S.items():
        try:
            s = strat(nm, pool)
        except KeyError:
            continue
        m(f"SR{tag}{pool}", s["sharpe"])
        m(f"GSR{tag}{pool}", s["gross_sharpe"])
        m(f"Ret{tag}{pool}", s["annualized_return"], 1, pct=True)
        m(f"GRet{tag}{pool}", s["gross_ann_return"], 1, pct=True)
        m(f"Vol{tag}{pool}", s["annualized_vol"], 1, pct=True)
        m(f"DD{tag}{pool}", s["max_drawdown"], 1, pct=True)
        m(f"TO{tag}{pool}", s["avg_turnover"], 2)
        m(f"BE{tag}{pool}", s["breakeven_bps"], 2)
        m(f"Corr{tag}{pool}", s["market_corr"], 2)
o = strat("Original repo specification", "--")
m("SROrig", o["sharpe"])
m("GSROrig", o["gross_sharpe"])
m("TOOrig", o["avg_turnover"], 2)
m("DDOrig", o["max_drawdown"], 0, pct=True)
m("BEOrig", o["breakeven_bps"], 2)

for pool in ("A", "B"):
    for k, tag in (("ML: mode features", "MLMode"), ("ML: mode + regime features", "MLAll")):
        p = R["placebo"][pool][k]
        m(f"PlaceboMean{pool}", p["placebo_mean"])
        m(f"PlaceboNF{pool}", p["placebo_p95"])
        m(f"PlaceboP{tag}{pool}", p["p_value"], 3)
    for k, tag in (("Heuristic: shortest half-life", "Heur"), ("ML: mode features", "MLMode"),
                   ("ML: mode + regime features", "MLAll"), ("ML + regime + exposure overlay", "MLOver"),
                   ("Equal weight + exposure overlay", "EWOver")):
        x = R["sharpe_diff"][pool][k]
        m(f"Diff{tag}{pool}", x["diff"], sign=True)
        m(f"DiffP{tag}{pool}", x["p"], 2)
    m(f"MeanExp{pool}", R["overlay"][pool]["mean_exposure"], 2)
    m(f"MinExp{pool}", R["overlay"][pool]["min_exposure"], 2)
    m(f"MeanExpEW{pool}", R["overlay_eq"][pool]["mean_exposure"], 2)

for r in R["regime_perf"]:
    tag = {"All candidates, equal weight": "EW", "ML: mode + regime features": "MLAll",
           "ML + regime + exposure overlay": "MLOver", "Equal weight + exposure overlay": "EWOver"}[r["strategy"]]
    m(f"RegSR{tag}{r['pool']}{'Stress' if r['stress'] else 'Calm'}", r["sharpe"])

for c in R.get("costs", []):
    if c["cost_bps"] in (0, 5, 10):
        tag = {"All candidates, equal weight": "EW", "Heuristic: shortest half-life": "Heur",
               "ML: mode + regime features": "MLAll"}[c["strategy"]]
        cw = {0: "Zero", 5: "Five", 10: "Ten"}[c["cost_bps"]]
        m(f"SRCost{cw}{tag}{c['pool']}", c["sharpe"])

for r in R.get("robustness", []):
    key = (r["test"].replace("\\", "").replace("$", "").replace("=", "").replace(" ", "")
           .replace("+", "p").replace("-", "").replace("%", "pct"))
    words = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six",
                                     "Seven", "Eight", "Nine"]))
    key = "".join(words.get(ch, ch) for ch in key if ch.isalnum())
    m(f"Rob{key}ML{r['pool']}", r["ml_sharpe"])
    m(f"Rob{key}EW{r['pool']}", r["eq_sharpe"])

lines = ["% Auto-generated by scripts/make_macros.py -- do not edit."]
for k, v in macros.items():
    lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
(ROOT / "paper" / "numbers.tex").write_text("\n".join(lines) + "\n")
print(f"wrote {len(macros)} macros")
