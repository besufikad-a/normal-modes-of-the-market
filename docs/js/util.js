// Shared helpers for the interactive page.
const C = {
  bg: "#0e1116", panel: "#151a22", line: "#263040", fg: "#e5e7eb", muted: "#9ca3af",
  stock: "#60a5fa", bond: "#34d399", gold: "#fbbf24", purple: "#c084fc", orange: "#fb923c",
  red: "#f87171", cyan: "#22d3ee",
  modes: ["#60a5fa", "#f87171", "#34d399", "#fbbf24", "#c084fc", "#22d3ee", "#fb923c", "#f472b6",
          "#a3e635", "#e879f9", "#38bdf8", "#facc15"],
};
const SECTOR_COLOR = { Tech: C.stock, Fin: C.gold, Energy: C.orange, Health: C.bond, Staples: C.purple };
const CORR_SCALE = [[0, "#2563eb"], [0.5, "#4b5563"], [1, "#ef4444"]];

const DATA = {};
async function loadData(name) {
  if (!DATA[name]) DATA[name] = fetch(`data/${name}.json`).then(r => r.json());
  return DATA[name];
}

function baseLayout(extra = {}) {
  return Object.assign({
    paper_bgcolor: C.panel, plot_bgcolor: C.panel,
    font: { color: C.fg, family: "-apple-system, Helvetica Neue, sans-serif", size: 13 },
    margin: { l: 60, r: 20, t: 30, b: 45 },
    xaxis: { gridcolor: C.line, zerolinecolor: C.line, linecolor: C.line },
    yaxis: { gridcolor: C.line, zerolinecolor: C.line, linecolor: C.line },
    legend: { orientation: "h", y: 1.08, x: 0, bgcolor: "rgba(0,0,0,0)" },
    hoverlabel: { bgcolor: "#111827", bordercolor: C.line, font: { color: C.fg } },
  }, extra);
}
const PCFG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

function axis3d(title, extra = {}) {
  return Object.assign({ title: { text: title, font: { color: C.muted } }, gridcolor: C.line,
    zerolinecolor: C.line, backgroundcolor: C.panel, showbackground: true, color: C.muted }, extra);
}

// Rolling Pearson correlation over a trailing window (NaN until full).
function rollingCorr(a, b, w) {
  const n = a.length, out = new Array(n).fill(null);
  for (let t = w - 1; t < n; t++) {
    let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    for (let i = t - w + 1; i <= t; i++) {
      sa += a[i]; sb += b[i]; saa += a[i] * a[i]; sbb += b[i] * b[i]; sab += a[i] * b[i];
    }
    const ca = saa - sa * sa / w, cb = sbb - sb * sb / w, cab = sab - sa * sb / w;
    out[t] = cab / Math.sqrt(ca * cb);
  }
  return out;
}

function cumGrowth(r) {
  let v = 1; return r.map(x => (v *= 1 + x));
}

function mean(a) { return a.reduce((s, x) => s + x, 0) / a.length; }
function std(a) { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); }

// Seeded RNG (mulberry32) + Box–Muller normals.
function rng(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function normals(n, r) {
  const out = [];
  for (let i = 0; i < n; i += 2) {
    const u = Math.max(r(), 1e-12), v = r();
    const m = Math.sqrt(-2 * Math.log(u));
    out.push(m * Math.cos(2 * Math.PI * v), m * Math.sin(2 * Math.PI * v));
  }
  return out.slice(0, n);
}

// Canvas helper: scale for device pixel ratio once.
function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.width, h = cv.height;
  if (!cv.dataset.scaled) {
    cv.style.aspectRatio = `${w} / ${h}`;
    cv.width = w * dpr; cv.height = h * dpr;
    cv.dataset.scaled = "1"; cv.dataset.w = w; cv.dataset.h = h;
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: +cv.dataset.w, h: +cv.dataset.h };
}

function drawSpring(ctx, x1, x2, y, coils = 8, amp = 9, color = C.muted, width = 2) {
  const L = x2 - x1, lead = 0.1 * L, m = 2 * coils;
  ctx.beginPath(); ctx.moveTo(x1, y); ctx.lineTo(x1 + lead, y);
  for (let j = 1; j < m; j++) ctx.lineTo(x1 + lead + (L - 2 * lead) * j / m, y + (j % 2 ? -amp : amp));
  ctx.lineTo(x2 - lead, y); ctx.lineTo(x2, y);
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
}

// Only animate canvases that are on screen.
function whenVisible(el, onShow, onHide) {
  new IntersectionObserver(es => es.forEach(e => (e.isIntersecting ? onShow() : onHide && onHide())),
    { threshold: 0.05 }).observe(el);
}

function fmt(x, nd = 2) { return x == null || Number.isNaN(x) ? "–" : x.toFixed(nd); }
function pct(x, nd = 1) { return x == null || Number.isNaN(x) ? "–" : (100 * x).toFixed(nd) + "%"; }
