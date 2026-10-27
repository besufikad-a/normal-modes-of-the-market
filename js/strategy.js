// Section 7: strategy lab — net PnL at any cost, recomputed client-side.
const STRAT_COLOR = {
  "All candidates, equal weight": C.muted, "MP signal modes, equal weight": C.red,
  "Heuristic: shortest half-life": C.orange, "ML: mode features": C.stock,
  "ML: mode + regime features": C.bond, "ML + regime + exposure overlay": C.purple,
  "Equal weight + exposure overlay": C.cyan, "Original repo specification": "#f472b6",
};
const DEFAULT_ON = new Set(["B|All candidates, equal weight", "B|ML: mode + regime features",
  "B|ML + regime + exposure overlay", "B|Equal weight + exposure overlay"]);

function perf(net, gross, to, stress) {
  const n = net.length, m = mean(net), s = std(net);
  let cum = 0, peak = 0, dd = 0;
  const curve = net.map(x => { cum += x; peak = Math.max(peak, cum); dd = Math.min(dd, cum - peak); return cum; });
  const gm = mean(gross), tm = mean(to);
  const sub = flag => { const x = net.filter((_, i) => stress[i] === flag); return x.length > 20 ? mean(x) / std(x) * Math.sqrt(252) : NaN; };
  return { curve, ret: m * 252, vol: s * Math.sqrt(252), sharpe: m / s * Math.sqrt(252), dd,
    gsharpe: gm / std(gross) * Math.sqrt(252), to: tm, be: gm / tm * 1e4, calm: sub(false), stress: sub(true) };
}

async function initStrategy() {
  const D = await loadData("strategies");
  const costS = document.getElementById("lab-cost"), costO = document.getElementById("lab-cost-o");
  const regBox = document.getElementById("lab-regime"), origBox = document.getElementById("lab-orig");
  const checks = document.getElementById("lab-checks");
  const series = D.series.filter(s => s.pool !== "--");
  const orig = D.series.find(s => s.pool === "--");
  const stress = D.p_stress.map((p, i) => (i > 0 ? (D.p_stress[i - 1] ?? 0) > 0.5 : false));

  const poolName = { A: "Pool A — modes 1–30 (pre-specified)", B: "Pool B — full spectrum" };
  checks.innerHTML = ["A", "B"].map(p => `<div class="pool">${poolName[p]}</div>` + series.filter(s => s.pool === p).map(s => {
    const id = `${s.pool}|${s.name}`;
    return `<label><input type="checkbox" value="${id}" ${DEFAULT_ON.has(id) ? "checked" : ""}>
      <span class="sw" style="background:${STRAT_COLOR[s.name]};${s.pool === "A" ? "opacity:.6" : ""}"></span>${s.name}</label>`;
  }).join("")).join("");

  // Stress-regime shading as rectangles over contiguous runs.
  const runs = []; let st = null;
  stress.forEach((f, i) => { if (f && st === null) st = i; if ((!f || i === stress.length - 1) && st !== null) { runs.push([st, i]); st = null; } });

  function draw() {
    const c = +costS.value; costO.textContent = c.toFixed(2).replace(/\.?0+$/, "") || "0";
    const on = [...checks.querySelectorAll("input:checked")].map(b => b.value);
    const list = series.filter(s => on.includes(`${s.pool}|${s.name}`));
    if (origBox.checked) list.push(orig);
    const traces = [], rows = [];
    for (const s of list) {
      const net = s.gross.map((g, i) => g - s.turnover[i] * c / 1e4);
      const p = perf(net, s.gross, s.turnover, stress);
      traces.push({ x: D.dates, y: p.curve.map(v => 100 * v), name: `${s.pool === "--" ? "" : s.pool + ": "}${s.name}`,
        line: { color: STRAT_COLOR[s.name], width: s.pool === "A" ? 1.4 : 2.2, dash: s.pool === "A" ? "dot" : "solid" },
        hovertemplate: "%{y:.1f}%<extra>%{fullData.name}</extra>" });
      rows.push([s, p]);
    }
    const shapes = regBox.checked ? runs.map(([a, b]) => ({ type: "rect", xref: "x", yref: "paper", x0: D.dates[a], x1: D.dates[b],
      y0: 0, y1: 1, fillcolor: "rgba(248,113,113,0.10)", line: { width: 0 } })) : [];
    shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: C.muted, width: 1 } });
    Plotly.react("lab-plot", traces, baseLayout({
      yaxis: { title: `cumulative net return (%, per unit gross) at ${c} bp`, gridcolor: C.line },
      xaxis: { gridcolor: C.line, type: "date" }, shapes, hovermode: "x unified",
      legend: { orientation: "h", y: -0.12 }, margin: { l: 65, r: 15, t: 20, b: 40 } }), PCFG);

    const tbl = document.getElementById("lab-table");
    const cls = x => (x > 0 ? "pos" : x < 0 ? "neg" : "");
    tbl.innerHTML = `<tr><th>Strategy</th><th>Pool</th><th>Net Sharpe</th><th>Gross Sharpe</th><th>Ann. return</th>
      <th>Ann. vol</th><th>Max DD</th><th>Turnover/day</th><th>Break-even (bp)</th><th>SR calm</th><th>SR stress</th></tr>` +
      rows.map(([s, p]) => `<tr><td><span class="dot" style="background:${STRAT_COLOR[s.name]}"></span>${s.name}</td>
        <td>${s.pool === "--" ? "–" : s.pool}</td><td class="${cls(p.sharpe)}"><b>${fmt(p.sharpe)}</b></td><td>${fmt(p.gsharpe)}</td>
        <td class="${cls(p.ret)}">${pct(p.ret)}</td><td>${pct(p.vol)}</td><td>${pct(p.dd)}</td><td>${fmt(p.to)}</td>
        <td>${fmt(p.be)}</td><td class="${cls(p.calm)}">${fmt(p.calm)}</td><td class="${cls(p.stress)}">${fmt(p.stress)}</td></tr>`).join("");
  }
  [costS, regBox, origBox].forEach(el => el.addEventListener("input", draw));
  checks.addEventListener("change", draw);
  draw();
}
