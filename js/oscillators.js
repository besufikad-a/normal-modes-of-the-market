// Section 2: two coupled oscillators, and the 2x2 correlation analogue.
function initOscillators() {
  const cv = document.getElementById("osc-canvas"), tv = document.getElementById("osc-trace");
  const kS = document.getElementById("osc-k"), kcS = document.getElementById("osc-kc"), gS = document.getElementById("osc-g");
  const read = document.getElementById("osc-read"), pauseBtn = document.getElementById("osc-pause");
  const S = { x: [1, 0], v: [0, 0], t: 0, running: true, visible: false, drag: -1, hist: [] };
  const R2 = Math.SQRT1_2;
  const params = () => ({ k: +kS.value, kc: +kcS.value, g: +gS.value });

  function accel(x, v, p) {
    return [-(p.k + p.kc) * x[0] + p.kc * x[1] - p.g * v[0], p.kc * x[0] - (p.k + p.kc) * x[1] - p.g * v[1]];
  }
  function step(dt) {
    const p = params(), nsub = Math.max(4, Math.ceil(dt / 0.005));
    for (let s = 0; s < nsub; s++) {
      const h = dt / nsub;
      if (S.drag >= 0) { S.v[S.drag] = 0; }
      const a = accel(S.x, S.v, p);
      for (let i = 0; i < 2; i++) { if (i === S.drag) continue; S.v[i] += a[i] * h; S.x[i] += S.v[i] * h; }
    }
    S.t += dt;
    const q = [R2 * (S.x[0] + S.x[1]), R2 * (S.x[0] - S.x[1])];
    S.hist.push({ t: S.t, x0: S.x[0], x1: S.x[1], q0: q[0], q1: q[1] });
    if (S.hist.length > 900) S.hist.shift();
  }

  const L = { wl: 50, wr: 850, e0: 320, e1: 580, sc: 95, y: 125, m: 64 };
  function drawSystem() {
    const { ctx, w, h } = setupCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const p0 = L.e0 + L.sc * S.x[0], p1 = L.e1 + L.sc * S.x[1];
    ctx.strokeStyle = C.muted; ctx.lineWidth = 5;
    for (const xw of [L.wl, L.wr]) { ctx.beginPath(); ctx.moveTo(xw, L.y - 50); ctx.lineTo(xw, L.y + 50); ctx.stroke(); }
    ctx.lineWidth = 1.5;
    for (let j = 0; j < 7; j++) {
      ctx.beginPath(); ctx.moveTo(L.wl, L.y - 45 + j * 15); ctx.lineTo(L.wl - 14, L.y - 33 + j * 15); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(L.wr, L.y - 45 + j * 15); ctx.lineTo(L.wr + 14, L.y - 57 + j * 15); ctx.stroke();
    }
    drawSpring(ctx, L.wl, p0 - L.m / 2, L.y);
    drawSpring(ctx, p0 + L.m / 2, p1 - L.m / 2, L.y, 10, 9, C.purple, 2.5);
    drawSpring(ctx, p1 + L.m / 2, L.wr, L.y);
    for (const [p, c, nm, e] of [[p0, C.stock, "Stocks", L.e0], [p1, C.bond, "Bonds", L.e1]]) {
      ctx.setLineDash([3, 4]); ctx.strokeStyle = "rgba(156,163,175,.4)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(e, L.y - 62); ctx.lineTo(e, L.y + 62); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = c; ctx.fillRect(p - L.m / 2, L.y - L.m / 2, L.m, L.m);
      ctx.fillStyle = C.bg; ctx.font = "600 16px Helvetica Neue, sans-serif"; ctx.textAlign = "center";
      ctx.fillText(nm, p, L.y + 5);
    }
    ctx.fillStyle = C.muted; ctx.font = "18px Helvetica Neue, sans-serif";
    ctx.fillText("k", (L.wl + L.e0) / 2 - 20, L.y - 26); ctx.fillText("k", (L.e1 + L.wr) / 2 + 20, L.y - 26);
    ctx.fillStyle = C.purple; ctx.fillText("coupling k꜀", (L.e0 + L.e1) / 2, L.y - 30);
    ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.font = "15px Helvetica Neue, sans-serif";
    ctx.fillText("drag a mass to displace it", 14, h - 12);
  }

  function drawTrace() {
    const { ctx, w, h } = setupCanvas(tv);
    ctx.clearRect(0, 0, w, h);
    const span = 30, tmax = S.t, tmin = tmax - span;
    const bands = [{ y0: 20, y1: 140, keys: [["x0", C.stock, "stocks x₁"], ["x1", C.bond, "bonds x₂"]], dash: [] },
                   { y0: 165, y1: 285, keys: [["q0", C.stock, "mode 1 q₁ (in phase)"], ["q1", C.red, "mode 2 q₂ (out of phase)"]], dash: [6, 4] }];
    for (const b of bands) {
      const mid = (b.y0 + b.y1) / 2, amp = (b.y1 - b.y0) / 2 / 1.6;
      ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(40, mid); ctx.lineTo(w - 10, mid); ctx.stroke();
      b.keys.forEach(([k, c, lab], j) => {
        ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.setLineDash(b.dash); ctx.beginPath();
        let first = true;
        for (const s of S.hist) {
          if (s.t < tmin) continue;
          const px = 40 + (s.t - tmin) / span * (w - 50), py = mid - s[k] * amp;
          if (first) { ctx.moveTo(px, py); first = false; } else ctx.lineTo(px, py);
        }
        ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = c; ctx.font = "15px Helvetica Neue, sans-serif"; ctx.textAlign = "left";
        ctx.fillText(lab, 48 + j * 230, b.y0 + 4);
      });
    }
    ctx.fillStyle = C.muted; ctx.textAlign = "right"; ctx.fillText("time →", w - 12, h - 6);
  }

  function readout() {
    const p = params(), w1 = Math.sqrt(p.k), w2 = Math.sqrt(p.k + 2 * p.kc);
    const q = [R2 * (S.x[0] + S.x[1]), R2 * (S.x[0] - S.x[1])], qd = [R2 * (S.v[0] + S.v[1]), R2 * (S.v[0] - S.v[1])];
    const E = [0.5 * (qd[0] ** 2 + w1 * w1 * q[0] ** 2), 0.5 * (qd[1] ** 2 + w2 * w2 * q[1] ** 2)];
    const tot = E[0] + E[1] || 1;
    document.getElementById("osc-k-o").textContent = p.k.toFixed(2);
    document.getElementById("osc-kc-o").textContent = p.kc.toFixed(2);
    document.getElementById("osc-g-o").textContent = p.g.toFixed(2);
    read.innerHTML = `<b class="c-stock">ω₁ = √k = ${w1.toFixed(3)}</b> &nbsp;(period ${(2 * Math.PI / w1).toFixed(1)})<br>
      <b class="c-red">ω₂ = √(k+2k꜀) = ${w2.toFixed(3)}</b> &nbsp;(period ${(2 * Math.PI / w2).toFixed(1)})<br>
      energy in mode 1: <b>${pct(E[0] / tot, 0)}</b> · mode 2: <b>${pct(E[1] / tot, 0)}</b><br>
      beat period: <b>${p.kc > 0 ? (2 * Math.PI / (w2 - w1)).toFixed(1) : "∞"}</b>`;
  }

  function set(x0, x1) { S.x = [x0, x1]; S.v = [0, 0]; }
  document.querySelectorAll("[data-osc]").forEach(b => b.addEventListener("click", () => {
    const a = b.dataset.osc;
    if (a === "mode1") set(0.9, 0.9);
    if (a === "mode2") set(0.9, -0.9);
    if (a === "push") set(1.2, 0);
    if (a === "pause") { S.running = !S.running; pauseBtn.textContent = S.running ? "Pause" : "Play"; }
  }));

  function localX(e) { const r = cv.getBoundingClientRect(); return (e.clientX - r.left) / r.width * 900; }
  cv.addEventListener("pointerdown", e => {
    const x = localX(e);
    const p0 = L.e0 + L.sc * S.x[0], p1 = L.e1 + L.sc * S.x[1];
    S.drag = Math.abs(x - p0) < 45 ? 0 : Math.abs(x - p1) < 45 ? 1 : -1;
    if (S.drag >= 0) cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener("pointermove", e => {
    if (S.drag < 0) return;
    const e0 = S.drag ? L.e1 : L.e0;
    S.x[S.drag] = Math.max(-1.6, Math.min(1.6, (localX(e) - e0) / L.sc));
  });
  cv.addEventListener("pointerup", () => { S.drag = -1; });

  let last = null;
  function frame(ts) {
    if (!S.visible) { last = null; return; }
    const dt = last == null ? 1 / 60 : Math.min(0.5, (ts - last) / 1000);
    last = ts;
    if (S.running) {  // record the trace at a fixed rate regardless of frame rate
      const n = Math.max(1, Math.round(dt * 60));
      for (let j = 0; j < n; j++) step(dt * 2.2 / n);
    }
    drawSystem(); drawTrace(); readout();
    requestAnimationFrame(frame);
  }
  whenVisible(cv, () => { if (!S.visible) { S.visible = true; requestAnimationFrame(frame); } }, () => { S.visible = false; });
  [kS, kcS, gS].forEach(s => s.addEventListener("input", readout));
  drawSystem(); drawTrace(); readout();
  initRho();
}

async function initRho() {
  const H = await loadData("history");
  const rhoS = document.getElementById("rho"), rhoO = document.getElementById("rho-o");
  const read = document.getElementById("rho-read"), play = document.getElementById("rho-play");
  const r = rng(11), n = 600, z1 = normals(n, r), z2 = normals(n, r);
  const hist = rollingCorr(H.r_stock, H.r_bond, 36);
  const hi = hist.map((v, i) => (v == null ? -1 : i)).filter(i => i >= 0);

  function draw(rho, label) {
    rhoO.textContent = rho.toFixed(2);
    const a = Math.sqrt((1 + rho) / 2), b = Math.sqrt((1 - rho) / 2);
    // x = a*u + b*w, y = a*u - b*w with u,w iid N(0,1) gives corr rho.
    const x = z1.map((u, i) => a * u + b * z2[i]), y = z1.map((u, i) => a * u - b * z2[i]);
    const l1 = 1 + rho, l2 = 1 - rho, s1 = 2 * Math.sqrt(l1) * Math.SQRT1_2, s2 = 2 * Math.sqrt(l2) * Math.SQRT1_2;
    const traces = [
      { x, y, mode: "markers", type: "scattergl", marker: { size: 4, color: rho > 0 ? "rgba(248,113,113,.45)" : "rgba(96,165,250,.45)" },
        hoverinfo: "skip", showlegend: false },
      { x: [-s1, s1], y: [-s1, s1], mode: "lines", line: { color: C.stock, width: 5 }, name: "mode 1 (1,1)/√2" },
      { x: [-s2, s2], y: [s2, -s2], mode: "lines", line: { color: C.red, width: 5 }, name: "mode 2 (1,−1)/√2" },
    ];
    Plotly.react("rho-plot", traces, baseLayout({
      xaxis: { range: [-4, 4], title: "asset 1 (standardized)", gridcolor: C.line, zeroline: false },
      yaxis: { range: [-4, 4], title: "asset 2", gridcolor: C.line, zeroline: false, scaleanchor: "x" },
      margin: { l: 55, r: 15, t: 40, b: 45 },
    }), PCFG);
    Plotly.react("rho-bars", [{ type: "bar", x: ["λ₁ = 1 + ρ", "λ₂ = 1 − ρ"], y: [l1, l2],
      marker: { color: [C.stock, C.red] }, text: [l1.toFixed(2), l2.toFixed(2)], textposition: "outside" }],
      baseLayout({ yaxis: { range: [0, 2.2], gridcolor: C.line, title: "variance" }, margin: { l: 50, r: 10, t: 15, b: 35 } }), PCFG);
    read.innerHTML = (label ? `<b>${label}</b> · ` : "") + `ρ = <b>${rho.toFixed(2)}</b> → ` +
      (rho >= 0 ? `<span class="c-stock">in-phase (market) mode dominates</span>` : `<span class="c-red">out-of-phase (hedge) mode dominates</span>`);
  }
  rhoS.addEventListener("input", () => draw(+rhoS.value));
  let timer = null;
  play.addEventListener("click", () => {
    if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ Replay stocks–bonds 1978–2026"; return; }
    let j = 0; play.textContent = "■ Stop";
    timer = setInterval(() => {
      const i = hi[j]; rhoS.value = hist[i]; draw(hist[i], H.date[i].slice(0, 4));
      j += 2; if (j >= hi.length) { clearInterval(timer); timer = null; play.textContent = "▶ Replay stocks–bonds 1978–2026"; }
    }, 40);
  });
  draw(+rhoS.value);
}
