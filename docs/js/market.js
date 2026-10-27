// Section 4: the market's own modes (12 stocks).
const MODE_DESC = {
  0: "Mode 0 — everything moves together: the in-phase 'market' mode.",
  1: "Mode 1 — defensives & tech vs. banks & energy: a risk-on / risk-off spread.",
  2: "Mode 2 — tech vs. defensives: growth vs. value.",
  3: "Mode 3 — energy vs. financials: a sector rotation.",
  11: "Mode 11 — JPM vs. BAC: the stiffest, lowest-variance mode, almost a pure pair trade.",
};

async function initMarket() {
  const M = await loadData("market");
  const n = M.names.length, V = M.eigenvectors, lam = M.eigenvalues;
  const lp = (1 + Math.sqrt(n / M.T)) ** 2;
  const scol = M.sectors.map(s => SECTOR_COLOR[s]);
  const tick = M.names.map((nm, i) => `<span style="color:${scol[i]}">${nm}</span>`);

  Plotly.react("mk-heat", [{ type: "heatmap", z: M.corr, x: tick, y: tick, colorscale: CORR_SCALE, zmin: -1, zmax: 1,
    hovertemplate: "%{y} · %{x}<br>ρ = %{z:.2f}<extra></extra>", colorbar: { thickness: 12, tickfont: { color: C.muted } } }],
    baseLayout({ title: { text: "correlation matrix", font: { size: 14, color: C.muted } },
      yaxis: { autorange: "reversed", gridcolor: C.line }, margin: { l: 60, r: 10, t: 40, b: 60 } }), PCFG);

  Plotly.react("mk-eig", [{ type: "bar", x: lam.map((_, k) => k), y: lam,
    marker: { color: lam.map(l => (l > lp ? C.red : C.muted)) }, hovertemplate: "mode %{x}: λ = %{y:.2f}<extra></extra>" }],
    baseLayout({ title: { text: "eigenvalues (variance of each mode)", font: { size: 14, color: C.muted } },
      xaxis: { title: "mode k", dtick: 1, gridcolor: C.line }, yaxis: { title: "λ_k", gridcolor: C.line },
      shapes: [{ type: "line", x0: -0.5, x1: n - 0.5, y0: lp, y1: lp, line: { color: C.cyan, dash: "dash", width: 2 } }],
      annotations: [{ x: n - 1, y: lp, text: `Marchenko–Pastur edge λ₊ = ${lp.toFixed(2)}`, showarrow: false,
        yanchor: "bottom", xanchor: "right", font: { color: C.cyan, size: 12 } },
        { x: 0, y: lam[0], text: `${(100 * lam[0] / n).toFixed(0)}% of all variance`, showarrow: true, ax: 95, ay: 25,
          font: { color: C.fg, size: 12 }, arrowcolor: C.muted }],
      margin: { l: 55, r: 15, t: 40, b: 45 } }), PCFG);

  // ---- each mode separately
  const box = document.getElementById("mk-modes"), desc = document.getElementById("mk-desc");
  box.innerHTML = lam.map((_, k) => `<button data-k="${k}">mode ${k}</button>`).join("");
  function showMode(k) {
    box.querySelectorAll("button").forEach(b => b.classList.toggle("on", +b.dataset.k === k));
    desc.textContent = (MODE_DESC[k] || `Mode ${k} — a higher-order spread.`) + `  λ = ${lam[k].toFixed(2)}` +
      (lam[k] > lp ? "  (above the noise floor)" : "  (inside the noise bulk)");
    const v = V.map(r => r[k]);
    Plotly.react("mk-vec", [{ type: "bar", x: M.names, y: v, marker: { color: v.map(x => (x >= 0 ? C.stock : C.red)),
      line: { color: scol, width: 2 } }, hovertemplate: "%{x}: %{y:.3f}<extra></extra>" }],
      baseLayout({ yaxis: { range: [-0.8, 0.8], title: "weight v_ik", gridcolor: C.line },
        xaxis: { tickfont: { color: C.muted } }, margin: { l: 55, r: 10, t: 15, b: 40 } }), PCFG);
  }
  box.addEventListener("click", e => { if (e.target.dataset.k) showMode(+e.target.dataset.k); });
  showMode(0);

  // ---- combination: rebuild a day
  const dayS = document.getElementById("mk-day"), dayO = document.getElementById("mk-day-o");
  const kS = document.getElementById("mk-k"), kO = document.getElementById("mk-k-o");
  dayS.max = M.dates.length - 1;
  const crash = M.dates.indexOf("2020-03-16");
  dayS.value = crash;

  function rebuild() {
    const d = +dayS.value, K = +kS.value, z = M.z[d];
    dayO.textContent = M.dates[d]; kO.textContent = K;
    const a = lam.map((_, k) => V.reduce((s, row, i) => s + row[k] * z[i], 0));
    const part = z.map((_, i) => a.slice(0, K).reduce((s, ak, k) => s + ak * V[i][k], 0));
    const tot = a.reduce((s, x) => s + x * x, 0);
    const exp = a.slice(0, K).reduce((s, x) => s + x * x, 0) / tot;
    Plotly.react("mk-rebuild", [
      { type: "bar", x: M.names, y: z, name: "actual", marker: { color: "rgba(0,0,0,0)", line: { color: C.fg, width: 2 } } },
      { type: "bar", x: M.names, y: part, name: `sum of first ${K} mode${K > 1 ? "s" : ""}`,
        marker: { color: part.map(x => (x >= 0 ? C.stock : C.red)), opacity: 0.85 } }],
      baseLayout({ barmode: "overlay", yaxis: { title: "standardized return", gridcolor: C.line },
        title: { text: `${M.dates[d]}: ${pct(exp, 0)} of the day's cross-section explained by ${K} mode${K > 1 ? "s" : ""}`,
          font: { size: 14 } }, legend: { orientation: "h", y: -0.12 }, margin: { l: 55, r: 10, t: 50, b: 60 } }), PCFG);
    Plotly.react("mk-share", [{ type: "bar", x: a.map((_, k) => `${k}`), y: a.map(x => (x * x) / tot),
      marker: { color: a.map((_, k) => (k < K ? C.modes[k % 12] : C.line)) }, hovertemplate: "mode %{x}: %{y:.1%}<extra></extra>" }],
      baseLayout({ title: { text: "share of that day carried by each mode  (a_k² / Σa²)", font: { size: 14, color: C.muted } },
        xaxis: { title: "mode k", gridcolor: C.line }, yaxis: { tickformat: ".0%", gridcolor: C.line },
        margin: { l: 55, r: 10, t: 45, b: 45 } }), PCFG);
  }
  dayS.addEventListener("input", rebuild); kS.addEventListener("input", rebuild);
  document.getElementById("mk-crash").addEventListener("click", () => { dayS.value = crash; rebuild(); });
  document.getElementById("mk-calm").addEventListener("click", () => {
    const calm = M.z.map((z, i) => [mean(z.map(Math.abs)), i]).filter(([m]) => m < 0.6);
    dayS.value = calm[Math.floor(Math.random() * calm.length)][1]; rebuild();
  });
  rebuild();
}
