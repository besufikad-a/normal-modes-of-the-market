// Section 3: N beads on a string — each mode separately, and combined.
function chainModes(N) {
  const omega = [], V = [];
  for (let k = 1; k <= N; k++) omega.push(2 * Math.sin(k * Math.PI / (2 * (N + 1))));
  for (let i = 1; i <= N; i++) {
    const row = [];
    for (let k = 1; k <= N; k++) row.push(Math.sin(i * k * Math.PI / (N + 1)) * Math.sqrt(2 / (N + 1)));
    V.push(row);
  }
  return { omega, V };
}

function initChain() {
  const nS = document.getElementById("ch-n"), nO = document.getElementById("ch-n-o");
  const modeSel = document.getElementById("ch-mode"), speedS = document.getElementById("ch-speed");
  const cv = document.getElementById("ch-canvas"), mv = document.getElementById("ch-multi");
  const S = { N: 5, M: chainModes(5), c: [], ph: [], t: 0, visible: false, seed: 3 };

  function randomize() {
    const r = rng(S.seed++), N = S.N, g = normals(N, r);
    S.c = g.map((x, k) => x / Math.sqrt(k + 1));
    S.ph = Array.from({ length: N }, () => r() * 2 * Math.PI);
    // Normalize so the sum fits on screen.
    let mx = 0;
    for (let i = 0; i < N; i++) { let s = 0; for (let k = 0; k < N; k++) s += Math.abs(S.M.V[i][k] * S.c[k]); mx = Math.max(mx, s); }
    S.c = S.c.map(x => x / mx);
  }
  function rebuild() {
    S.N = +nS.value; nO.textContent = S.N; S.M = chainModes(S.N);
    const cur = modeSel.value;
    modeSel.innerHTML = `<option value="all">All (combination)</option>` +
      S.M.omega.map((w, k) => `<option value="${k}">mode ${k + 1} — ω = ${w.toFixed(3)}</option>`).join("");
    modeSel.value = cur === "all" || +cur < S.N ? cur : "all";
    randomize(); draw3d();
  }

  function drawMain() {
    const { ctx, w, h } = setupCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const N = S.N, x0 = 50, x1 = w - 300, y = h / 2, A = 105;
    const xs = Array.from({ length: N }, (_, i) => x0 + (x1 - x0) * (i + 1) / (N + 1));
    const sel = modeSel.value;
    const q = S.M.omega.map((om, k) => (sel === "all" ? S.c[k] * Math.cos(om * S.t + S.ph[k]) : (k === +sel ? Math.cos(om * S.t) / Math.max(...S.M.V.map(r => Math.abs(r[k]))) : 0)));
    ctx.strokeStyle = C.line; ctx.setLineDash([4, 5]); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.setLineDash([]);
    if (sel === "all") {
      for (let k = 0; k < N; k++) {
        ctx.strokeStyle = C.modes[k % 12]; ctx.globalAlpha = 0.5; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(x0, y);
        xs.forEach((x, i) => ctx.lineTo(x, y - A * S.M.V[i][k] * q[k])); ctx.lineTo(x1, y); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    const ys = xs.map((_, i) => y - A * S.M.V[i].reduce((s, v, k) => s + v * q[k], 0));
    ctx.strokeStyle = C.fg; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(x0, y);
    xs.forEach((x, i) => ctx.lineTo(x, ys[i])); ctx.lineTo(x1, y); ctx.stroke();
    const bc = sel === "all" ? C.gold : C.modes[+sel % 12];
    xs.forEach((x, i) => { ctx.fillStyle = bc; ctx.beginPath(); ctx.arc(x, ys[i], N > 14 ? 5 : 8, 0, 7); ctx.fill(); });
    for (const xw of [x0, x1]) { ctx.fillStyle = C.muted; ctx.fillRect(xw - 3, y - 18, 6, 36); }
    // Modal amplitude bars.
    const bx0 = w - 260, bw = 240 / N;
    ctx.fillStyle = C.muted; ctx.font = "12px Helvetica Neue, sans-serif"; ctx.textAlign = "left";
    ctx.fillText("modal amplitudes q_k(t)", bx0, 22);
    ctx.strokeStyle = C.line; ctx.beginPath(); ctx.moveTo(bx0, y); ctx.lineTo(bx0 + 240, y); ctx.stroke();
    const qmax = Math.max(1e-9, ...q.map(Math.abs), ...(sel === "all" ? S.c.map(Math.abs) : [1]));
    q.forEach((v, k) => { ctx.fillStyle = C.modes[k % 12]; const hh = v / qmax * 100; ctx.fillRect(bx0 + k * bw + bw * 0.15, y - Math.max(0, hh), bw * 0.7, Math.abs(hh)); });
  }

  function drawMulti() {
    const { ctx, w, h } = setupCanvas(mv);
    ctx.clearRect(0, 0, w, h);
    const N = S.N, rows = N, rh = (h - 10) / rows, x0 = 30, x1 = w - 130;
    const xs = Array.from({ length: N }, (_, i) => x0 + (x1 - x0) * (i + 1) / (N + 1));
    for (let k = 0; k < N; k++) {
      const y = 5 + rh * (k + 0.5), A = rh * 0.4, c = Math.cos(S.M.omega[k] * S.t);
      const vmax = Math.max(...S.M.V.map(r => Math.abs(r[k])));
      ctx.strokeStyle = C.modes[k % 12]; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(x0, y);
      const ys = xs.map((_, i) => y - A * S.M.V[i][k] / vmax * c);
      xs.forEach((x, i) => ctx.lineTo(x, ys[i])); ctx.lineTo(x1, y); ctx.stroke();
      if (N <= 16) xs.forEach((x, i) => { ctx.fillStyle = C.modes[k % 12]; ctx.beginPath(); ctx.arc(x, ys[i], 3, 0, 7); ctx.fill(); });
      ctx.fillStyle = C.modes[k % 12]; ctx.font = `${Math.min(13, rh * 0.6)}px Helvetica Neue, sans-serif`; ctx.textAlign = "left";
      ctx.fillText(`mode ${k + 1}  ω=${S.M.omega[k].toFixed(2)}`, x1 + 10, y + 4);
    }
  }

  function draw3d() {
    const N = S.N, k = [...Array(N).keys()].map(i => i + 1);
    const z = S.M.V[0].map((_, kk) => S.M.V.map(r => r[kk]));  // z[mode][bead]
    Plotly.react("ch-3d", [{ type: "surface", x: k, y: k, z, colorscale: CORR_SCALE, showscale: false,
      contours: { z: { show: true, usecolormap: true, project: { z: false } } } }],
      baseLayout({ margin: { l: 0, r: 0, t: 0, b: 0 }, scene: {
        xaxis: axis3d("bead i"), yaxis: axis3d("mode k"), zaxis: axis3d("v_ik"),
        camera: { eye: { x: -1.5, y: -1.6, z: 0.9 } } } }), PCFG);
  }

  let last = null;
  function frame(ts) {
    if (!S.visible) { last = null; return; }
    const dt = last == null ? 1 / 60 : Math.min(0.5, (ts - last) / 1000); last = ts;
    S.t += dt * 2.4 * +speedS.value;
    drawMain(); drawMulti();
    requestAnimationFrame(frame);
  }
  whenVisible(cv, () => { if (!S.visible) { S.visible = true; requestAnimationFrame(frame); } }, () => { S.visible = false; });
  nS.addEventListener("input", rebuild);
  document.getElementById("ch-rand").addEventListener("click", () => { modeSel.value = "all"; randomize(); });
  modeSel.innerHTML = `<option value="all">All (combination)</option>`;
  rebuild(); drawMain(); drawMulti();
}
