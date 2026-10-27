// Page bootstrap: hero tiles, findings section, ML charts, and lazy init of each tool.
async function initFindings() {
  const R = await loadData("results");
  const b = R.spectrum_buckets;
  const get = (sig, name) => b.find(x => x.signal === sig && x.bucket.startsWith(name));
  const sig = get("level", "Signal"), mid = get("level", "Middle");
  const best = R.strategies.find(s => s.name === "ML + regime + exposure overlay" && s.pool === "B");
  const aucA = R.auc.find(a => a.pool === "A" && a.model === "logit" && a.features === "mode");
  document.getElementById("tiles").innerHTML = [
    [sig.sharpe.toFixed(2), "Sharpe of reversion in the Marchenko–Pastur 'signal' modes (t = " + sig.t_nw.toFixed(2) + ")"],
    [mid.sharpe.toFixed(2), "Sharpe in the low-variance noise bulk, modes 100–174 (t = " + mid.t_nw.toFixed(1) + ")"],
    [aucA.auc.toFixed(3), "out-of-sample AUC predicting which modes revert (0.5 = chance)"],
    [best.sharpe.toFixed(2), "best net Sharpe at 2 bp (ML + regime overlay; overlay chosen after seeing the data)"],
  ].map(([v, l]) => `<div class="tile"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");

  const labels = ["Market (k=0)", "Signal (above MP)", "Edge of bulk", "Upper bulk", "Middle bulk", "Lower bulk"];
  const prefixes = ["Market", "Signal", "Edge", "Upper", "Middle", "Lower"];
  function drawRev(s) {
    const rows = prefixes.map(p => get(s, p));
    Plotly.react("rev-plot", [{ type: "bar", x: labels, y: rows.map(r => r.sharpe),
      marker: { color: rows.map(r => (r.t_nw > 1.96 ? C.bond : r.t_nw > 1.64 ? C.gold : r.sharpe < 0 ? C.red : C.muted)) },
      text: rows.map(r => `t = ${(Math.abs(r.t_nw) < 0.05 ? 0 : r.t_nw).toFixed(1)}`), textposition: "outside", hovertemplate: "%{x}: Sharpe %{y:.2f}<extra></extra>" }],
      baseLayout({ yaxis: { title: "out-of-sample Sharpe, before costs", gridcolor: C.line, range: [-0.8, 1.35] },
        title: { text: "smaller eigenvalue →", x: 0.98, xanchor: "right", font: { size: 13, color: C.gold } },
        xaxis: { tickangle: -25 }, margin: { l: 60, r: 15, t: 40, b: 85 } }), PCFG);
    document.getElementById("rev-callout").innerHTML = s === "level"
      ? `The modes random-matrix theory certifies as <b>signal</b> do <b>not</b> revert (Sharpe ${sig.sharpe.toFixed(2)}).
         Reversion appears deep in the <b>noise bulk</b>, the low-variance directions the oscillator picture says are stiffest.
         It is mostly a one-day effect, and break-even costs are only ~2 bp.`
      : `The original repository traded a daily-return reversal on the top signal modes. On those modes the signal is
         <b>negative</b> (Sharpe ${get("return", "Signal").sharpe.toFixed(2)}): those modes trend rather than revert.`;
  }
  document.querySelectorAll("input[name=sig]").forEach(r => r.addEventListener("change", e => drawRev(e.target.value)));
  drawRev("level");

  // AUC with CIs.
  const models = [["logit", "Logistic"], ["gbm", "Gradient boosting"], ["rf", "Random forest"]];
  const tr = [];
  for (const [pool, col] of [["A", C.stock], ["B", C.bond]]) {
    for (const fs of ["mode", "all"]) {
      const rows = models.map(([m]) => R.auc.find(a => a.pool === pool && a.model === m && a.features === fs));
      tr.push({ type: "bar", name: `${pool === "A" ? "modes 1–30" : "full spectrum"} · ${fs === "mode" ? "mode features" : "+ regime"}`,
        x: models.map(m => m[1]), y: rows.map(r => r.auc),
        error_y: { type: "data", symmetric: false, array: rows.map(r => r.hi - r.auc), arrayminus: rows.map(r => r.auc - r.lo), color: C.muted },
        marker: { color: col, opacity: fs === "mode" ? 0.55 : 1 }, hovertemplate: "AUC %{y:.3f}<extra>%{fullData.name}</extra>" });
    }
  }
  Plotly.react("auc-plot", tr, baseLayout({ barmode: "group", yaxis: { range: [0.48, 0.56], title: "out-of-sample AUC", gridcolor: C.line },
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: { color: C.red, dash: "dot" } }],
    legend: { orientation: "h", y: -0.15, font: { size: 11 } }, margin: { l: 60, r: 10, t: 20, b: 40 } }), PCFG);

  const ab = R.ablation.filter(a => a.pool === "A");
  Plotly.react("abl-plot", [{ type: "bar", orientation: "h", y: ab.map(a => a.group), x: ab.map(a => a.solo_auc),
    marker: { color: ab.map(a => (a.solo_auc > 0.515 ? C.gold : C.muted)) }, hovertemplate: "%{y}: AUC %{x:.3f}<extra></extra>" }],
    baseLayout({ title: { text: "each feature group alone (modes 1–30)", font: { size: 13, color: C.muted } },
      xaxis: { range: [0.48, 0.53], title: "AUC", gridcolor: C.line }, yaxis: { autorange: "reversed" },
      shapes: [{ type: "line", x0: 0.5, x1: 0.5, yref: "paper", y0: 0, y1: 1, line: { color: C.red, dash: "dot" } }],
      margin: { l: 150, r: 15, t: 40, b: 45 } }), PCFG);
}

function lazy(id, fn) {
  const el = document.getElementById(id);
  let done = false;
  new IntersectionObserver((es, obs) => es.forEach(e => {
    if (e.isIntersecting && !done) { done = true; obs.disconnect(); fn(); }
  }), { rootMargin: "400px" }).observe(el);
}

window.addEventListener("DOMContentLoaded", () => {
  if (window.renderMathInElement) {
    renderMathInElement(document.body, { delimiters: [{ left: "\\(", right: "\\)", display: false }] });
  }
  initFindings();
  lazy("history", initHistory);
  lazy("two", initOscillators);
  lazy("chain", initChain);
  lazy("market", initMarket);
  lazy("spectrum", initSpectrum);
  lazy("lab", initStrategy);
});
