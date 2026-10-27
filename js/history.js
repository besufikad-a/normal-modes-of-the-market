// Section 1: fifty years of stocks, bonds and gold.
const EVENTS = [
  [1980.0, "Volcker"], [1987.8, "Black Monday"], [2000.2, "Dot-com peak"],
  [2008.7, "GFC"], [2020.2, "COVID"], [2022.4, "Inflation shock"],
];

async function initHistory() {
  const H = await loadData("history");
  const x = H.date.map(d => d + "-15");
  const ret = { stock: H.r_stock, bond: H.r_bond, gold: H.r_gold };
  const val = { stock: cumGrowth(H.r_stock), bond: cumGrowth(H.r_bond), gold: cumGrowth(H.r_gold) };
  const col = { stock: C.stock, bond: C.bond, gold: C.gold };
  const name = { stock: "Stocks", bond: "Bonds", gold: "Gold" };

  const win = document.getElementById("h-win"), winO = document.getElementById("h-win-o");
  const logBox = document.getElementById("h-log");
  const assetBoxes = [...document.querySelectorAll(".h-asset")];

  function draw() {
    const w = +win.value; winO.textContent = w;
    const on = assetBoxes.filter(b => b.checked).map(b => b.value);
    const traces = on.map(a => ({
      x, y: val[a], name: name[a], line: { color: col[a], width: 2 }, xaxis: "x", yaxis: "y",
      hovertemplate: `${name[a]}: $%{y:.2f}<extra>%{x|%b %Y}</extra>`,
    }));
    const pairs = [["stock", "bond", C.purple], ["stock", "gold", C.orange], ["bond", "gold", C.cyan]];
    for (const [a, b, c] of pairs) {
      traces.push({
        x, y: rollingCorr(ret[a], ret[b], w), name: `${name[a]} ↔ ${name[b]}`, xaxis: "x", yaxis: "y2",
        line: { color: c, width: 1.8 }, hovertemplate: `ρ = %{y:.2f}<extra>%{x|%b %Y}</extra>`,
      });
    }
    const shapes = [
      { type: "rect", xref: "x", yref: "y2", x0: "2000-01-01", x1: "2021-12-31", y0: -1, y1: 1,
        fillcolor: "rgba(96,165,250,0.08)", line: { width: 0 } },
      { type: "line", xref: "paper", x0: 0, x1: 1, yref: "y2", y0: 0, y1: 0, line: { color: C.muted, width: 1, dash: "dot" } },
      ...EVENTS.map(([t]) => ({ type: "line", xref: "x", yref: "paper", x0: yearToDate(t), x1: yearToDate(t),
        y0: 0, y1: 1, line: { color: "rgba(156,163,175,0.35)", width: 1, dash: "dot" } })),
    ];
    const ann = EVENTS.map(([t, s], j) => ({ x: yearToDate(t), y: j % 2 ? 0.975 : 1.0, xref: "x", yref: "paper", text: s, showarrow: false,
      font: { size: 11, color: C.muted }, yanchor: "bottom", textangle: -0 }));
    ann.push({ x: "2011-01-01", y: -0.92, xref: "x", yref: "y2", text: "bonds hedged stocks", showarrow: false,
      font: { size: 12, color: C.stock } });
    const layout = baseLayout({
      grid: { rows: 2, columns: 1, roworder: "top to bottom" },
      xaxis: { gridcolor: C.line, anchor: "y2", type: "date" },
      yaxis: { domain: [0.42, 0.95], type: logBox.checked ? "log" : "linear", title: "growth of $1",
               gridcolor: C.line, tickprefix: "$" },
      yaxis2: { domain: [0, 0.34], range: [-1, 1], title: `${w}-month correlation`, gridcolor: C.line, zeroline: false },
      shapes, annotations: ann, margin: { l: 70, r: 20, t: 40, b: 60 }, hovermode: "x unified",
      legend: { orientation: "h", y: -0.1 },
    });
    Plotly.react("h-plot", traces, layout, PCFG);
    draw3d(w);
  }

  const pairSel = document.getElementById("h3-pair");
  function draw3d(w) {
    const [a, b] = pairSel.value.split(",");
    const rho = rollingCorr(ret[a], ret[b], w);
    const idx = rho.map((v, i) => (v == null ? -1 : i)).filter(i => i >= 0);
    const tr = {
      type: "scatter3d", mode: "markers",
      x: idx.map(i => ret[a][i]), y: idx.map(i => ret[b][i]), z: idx.map(i => H.year[i]),
      text: idx.map(i => `${H.date[i]}<br>ρ = ${rho[i].toFixed(2)}`),
      hovertemplate: `%{text}<br>${name[a]} %{x:.1%}<br>${name[b]} %{y:.1%}<extra></extra>`,
      marker: { size: 3, color: idx.map(i => rho[i]), colorscale: CORR_SCALE, cmin: -0.8, cmax: 0.8,
                colorbar: { title: "ρ", thickness: 12, len: 0.6, tickfont: { color: C.muted } }, opacity: 0.9 },
    };
    // Decade principal axes.
    const lines = [];
    for (let dec = 1975; dec < 2026; dec += 10) {
      const ii = idx.filter(i => H.year[i] >= dec && H.year[i] < dec + 10);
      if (ii.length < 24) continue;
      const xa = ii.map(i => ret[a][i]), xb = ii.map(i => ret[b][i]);
      const ma = mean(xa), mb = mean(xb);
      let saa = 0, sbb = 0, sab = 0;
      for (let k = 0; k < ii.length; k++) { saa += (xa[k] - ma) ** 2; sbb += (xb[k] - mb) ** 2; sab += (xa[k] - ma) * (xb[k] - mb); }
      saa /= ii.length; sbb /= ii.length; sab /= ii.length;
      const tr2 = saa + sbb, det = saa * sbb - sab * sab, l1 = tr2 / 2 + Math.sqrt(tr2 * tr2 / 4 - det);
      let vx = sab, vy = l1 - saa; if (Math.abs(sab) < 1e-12) { vx = 1; vy = 0; }
      const nrm = Math.hypot(vx, vy), s = 2.2 * Math.sqrt(l1);
      vx = vx / nrm * s; vy = vy / nrm * s;
      const r = sab / Math.sqrt(saa * sbb), z = dec + 5;
      lines.push({ type: "scatter3d", mode: "lines+text", x: [-vx, vx], y: [-vy, vy], z: [z, z],
        line: { width: 7, color: r > 0 ? C.red : C.stock }, text: ["", `${dec}s ρ=${r.toFixed(2)}`],
        textfont: { color: r > 0 ? C.red : C.stock, size: 12 }, hoverinfo: "skip", showlegend: false });
    }
    const layout = baseLayout({
      margin: { l: 0, r: 0, t: 10, b: 0 }, showlegend: false,
      scene: {
        xaxis: axis3d(`${name[a]} return`, { tickformat: ".0%" }),
        yaxis: axis3d(`${name[b]} return`, { tickformat: ".0%" }),
        zaxis: axis3d("year"), aspectmode: "manual", aspectratio: { x: 1, y: 1, z: 1.3 },
        camera: { eye: { x: 1.6, y: -1.5, z: 0.7 } },
      },
    });
    Plotly.react("h3-plot", [tr, ...lines], layout, PCFG);
  }

  win.addEventListener("input", draw);
  logBox.addEventListener("change", draw);
  assetBoxes.forEach(b => b.addEventListener("change", draw));
  pairSel.addEventListener("change", () => draw3d(+win.value));
  draw();
}

function yearToDate(t) {
  const y = Math.floor(t), m = Math.min(12, Math.max(1, Math.round((t - y) * 12) + 1));
  return `${y}-${String(m).padStart(2, "0")}-01`;
}
