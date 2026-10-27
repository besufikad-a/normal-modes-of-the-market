// Section 5: the rolling spectrum of 417 stocks.
async function initSpectrum() {
  const S = await loadData("spectrum");
  const K = S.lam[0].length, ranks = [...Array(K).keys()];
  const dates = S.t.map(t => { const y = Math.floor(t), d = Math.round((t - y) * 365); return new Date(Date.UTC(y, 0, 1 + d)); });
  const lbl = dates.map(d => d.toISOString().slice(0, 7));
  const logL = S.lam.map(row => row.map(Math.log10));
  const lp = S.mp_upper;

  Plotly.react("sp-3d", [
    { type: "surface", x: ranks, y: lbl, z: logL, colorscale: [[0, "#1e3a8a"], [0.3, C.cyan], [0.55, C.gold], [1, C.red]],
      cmin: -0.2, cmax: 2.3, colorbar: { title: "log₁₀ λ", thickness: 12, tickfont: { color: C.muted } },
      hovertemplate: "%{y} · mode %{x}<br>λ = 10^%{z:.2f}<extra></extra>" },
    { type: "surface", x: ranks, y: lbl, z: logL.map(r => r.map(() => Math.log10(lp))), showscale: false, opacity: 0.35,
      colorscale: [[0, C.cyan], [1, C.cyan]], hoverinfo: "skip", name: "MP floor" },
  ], baseLayout({ margin: { l: 0, r: 0, t: 10, b: 0 }, scene: {
    xaxis: axis3d("mode rank k"), yaxis: axis3d("window end"), zaxis: axis3d("log₁₀ λ"),
    aspectmode: "manual", aspectratio: { x: 1.1, y: 1.5, z: 0.7 }, camera: { eye: { x: 1.7, y: -1.4, z: 0.9 } } } }), PCFG);

  const tS = document.getElementById("sp-t"), tO = document.getElementById("sp-t-o");
  tS.max = S.t.length - 1;
  const peak = S.lam.reduce((b, r, i) => (r[0] > S.lam[b][0] ? i : b), 0);
  tS.value = peak;
  function draw2d() {
    const i = +tS.value, lam = S.lam[i];
    tO.textContent = lbl[i];
    Plotly.react("sp-2d", [{ type: "bar", x: ranks, y: lam, marker: { color: lam.map(l => (l > lp ? C.red : C.muted)) },
      hovertemplate: "mode %{x}: λ = %{y:.2f}<extra></extra>" }],
      baseLayout({ yaxis: { type: "log", title: "λ_k (log)", gridcolor: C.line, range: [-0.3, 2.3] },
        xaxis: { title: "mode rank k (first 60 of 417)", gridcolor: C.line },
        shapes: [{ type: "line", x0: -0.5, x1: K - 0.5, y0: lp, y1: lp, line: { color: C.cyan, dash: "dash", width: 2 } }],
        title: { text: `252 days to ${lbl[i]}: ${S.n_signal[i]} modes above the noise floor · market mode carries ${(100 * lam[0] / S.N).toFixed(0)}% of variance`,
          font: { size: 14 } }, margin: { l: 60, r: 15, t: 45, b: 45 } }), PCFG);
  }
  tS.addEventListener("input", draw2d);
  draw2d();
}
