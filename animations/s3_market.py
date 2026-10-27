"""Scenes 7–9: the market as coupled oscillators.

S7_MarketModes   12 large stocks: correlation matrix, spectrum, each mode
                 separately, then a real day rebuilt as a sum of modes.
S8_Spectrum3D    3-D: rolling eigenvalue spectrum of 417 stocks
                 (time × mode rank × log λ) against the Marchenko–Pastur floor.
S9_Findings      the physics link Σ = (s²/2) K⁻¹ and the paper's result.
"""
from __future__ import annotations

import numpy as np
from manim import *

from common import (BG, BOND, CYAN, FG, GOLD, GRID, MUTED, ORANGE, POS, PURPLE, STOCK,
                    T, corr_color, load_json)

MM = load_json("market_modes.json")
SECTOR_COLOR = {"Tech": STOCK, "Fin": GOLD, "Energy": ORANGE, "Health": BOND, "Staples": PURPLE}


def bar_chart(values, colors, width=7.5, height=2.6, ymax=None, center=ORIGIN):
    n = len(values)
    ymax = ymax or max(abs(v) for v in values)
    bw = width / n * 0.7
    g = VGroup()
    for i, v in enumerate(values):
        h = v / ymax * height / 2
        r = Rectangle(width=bw, height=max(abs(h), 1e-3), fill_color=colors[i], fill_opacity=0.9, stroke_width=0)
        r.move_to(center + RIGHT * (-width / 2 + (i + 0.5) * width / n) + UP * h / 2)
        g.add(r)
    return g


class S7_MarketModes(Scene):
    def construct(self):
        self.camera.background_color = BG
        names, sectors = MM["names"], MM["sectors"]
        C = np.array(MM["corr"])
        lam = np.array(MM["eigenvalues"])
        V = np.array(MM["eigenvectors"])
        n = len(names)
        cols = [SECTOR_COLOR[s] for s in sectors]

        title = T("Now the market: 12 stocks, 2015 – 2024", 38, weight="BOLD").to_edge(UP, buff=0.3)
        self.play(FadeIn(title, shift=DOWN * 0.2))

        # ---------------- correlation heatmap
        cell = 0.36
        heat = VGroup()
        for i in range(n):
            for j in range(n):
                sq = Square(cell, stroke_width=0.5, stroke_color=BG, fill_opacity=1,
                            fill_color=corr_color(C[i, j] * 1.4 if i != j else 1))
                sq.move_to(np.array([(j - n / 2 + 0.5) * cell, (n / 2 - i - 0.5) * cell, 0]))
                heat.add(sq)
        heat.move_to(LEFT * 3.0 + DOWN * 0.5)
        rl = VGroup(*[T(nm, 14, cols[i]).next_to(heat[i * n], LEFT, buff=0.1) for i, nm in enumerate(names)])
        cl = VGroup(*[T(nm, 14, cols[j]).rotate(PI / 2).next_to(heat[j], UP, buff=0.08) for j, nm in enumerate(names)])
        hl = T("correlation matrix C", 20, MUTED).next_to(heat, DOWN, buff=0.25)
        self.play(LaggedStart(*[FadeIn(s) for s in heat], lag_ratio=0.004), FadeIn(rl), FadeIn(cl),
                  FadeIn(hl), run_time=2)
        legend = VGroup(*[VGroup(Square(0.18, fill_color=c, fill_opacity=1, stroke_width=0), T(s, 16, c))
                          .arrange(RIGHT, buff=0.08) for s, c in SECTOR_COLOR.items()]).arrange(RIGHT, buff=0.25)
        legend.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(legend))

        # ---------------- spectrum with MP edge
        T_obs = 2515
        lp = (1 + np.sqrt(n / T_obs)) ** 2
        ax = Axes(x_range=[0, n, 1], y_range=[0, 6, 1], x_length=5.2, y_length=3.6,
                  axis_config={"color": GRID}, tips=False).move_to(RIGHT * 3.4 + DOWN * 0.4)
        bars = VGroup(*[
            Rectangle(width=0.3, height=max(ax.y_length * l / 6, 1e-3), fill_opacity=0.9, stroke_width=0,
                      fill_color=POS if l > lp else MUTED).move_to(ax.c2p(k + 0.5, 0), aligned_edge=DOWN)
            for k, l in enumerate(lam)])
        mp = DashedLine(ax.c2p(0, lp), ax.c2p(n, lp), color=CYAN, stroke_width=2)
        mpl = T(f"Marchenko–Pastur edge λ₊ = {lp:.2f}", 16, CYAN).next_to(mp, UP, buff=0.05).align_to(mp, RIGHT)
        axl = VGroup(T("eigenvalue λ_k  (variance of mode k)", 16, MUTED).next_to(ax, UP, buff=0.1),
                     T("mode k →", 16, MUTED).next_to(ax, DOWN, buff=0.1))
        self.play(Create(ax), FadeIn(axl), run_time=1)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.08), run_time=1.6)
        self.play(Create(mp), FadeIn(mpl))
        note = T(f"mode 0 carries {lam[0] / n:.0%} of all variance; {int((lam > lp).sum())} modes clear the noise floor",
                 20, FG).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(note))
        self.wait(2)
        self.play(*[FadeOut(m) for m in (heat, rl, cl, hl, ax, bars, mp, mpl, axl, note)])

        # ---------------- each mode separately
        center = DOWN * 0.6
        base = Line(center + LEFT * 4, center + RIGHT * 4, color=GRID)
        nl = VGroup(*[T(nm, 15, cols[i]).rotate(PI / 4)
                      .move_to(center + RIGHT * (-4 + (i + 0.5) * 8 / n) + DOWN * 1.9) for i, nm in enumerate(names)])
        self.play(Create(base), FadeIn(nl))
        story = [
            (0, "Mode 0 — everything moves together", "the in-phase mode: 'the market'"),
            (1, "Mode 1 — defensives & tech vs. banks & energy", "a 'risk-on / risk-off' spread"),
            (2, "Mode 2 — tech vs. defensives", "growth vs. value"),
            (3, "Mode 3 — energy vs. financials", "a sector rotation"),
            (11, "Mode 11 — JPM vs. BAC", "the stiffest, lowest-variance mode: almost a pure pair trade"),
        ]
        cur = None
        heading = None
        for k, h1, h2 in story:
            vals = V[:, k]
            ch = bar_chart(vals, [STOCK if v > 0 else POS for v in vals], width=8, height=3.2, ymax=0.75,
                           center=center)
            hd = VGroup(T(h1, 26, GOLD, "BOLD"),
                        VGroup(T(h2, 20, MUTED), MathTex(rf"\lambda_{{{k}}} = {lam[k]:.2f}", font_size=30))
                        .arrange(RIGHT, buff=0.4)).arrange(DOWN, buff=0.12).next_to(legend, DOWN, buff=0.25)
            if cur is None:
                self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in ch], lag_ratio=0.05), FadeIn(hd), run_time=1.4)
            else:
                self.play(Transform(cur, ch), ReplacementTransform(heading, hd), run_time=1.2)
                ch = cur
            cur, heading = ch, hd
            self.wait(1.8)
        self.play(FadeOut(cur), FadeOut(heading))

        # ---------------- combination: rebuild a real day
        z = np.array(MM["day_z"])
        a = np.array(MM["day_coef"])
        ymax = np.abs(z).max() * 1.1
        target = VGroup(*[
            Rectangle(width=8 / n * 0.7, height=abs(v) / ymax * 1.6, stroke_color=FG, stroke_width=2,
                      fill_opacity=0).move_to(center + RIGHT * (-4 + (i + 0.5) * 8 / n) + UP * v / ymax * 0.8)
            for i, v in enumerate(z)])
        hd = VGroup(T(f"Combination — {MM['day']}, the COVID crash", 26, GOLD, "BOLD"),
                    MathTex(r"r_{\text{day}}=\sum_k a_k\, v_k,\qquad a_k = v_k^\top r_{\text{day}}", font_size=30)
                    ).arrange(DOWN, buff=0.12).next_to(legend, DOWN, buff=0.25)
        self.play(FadeIn(hd), Create(target), run_time=1.2)
        tlab = T("white outlines: actual standardized returns that day", 16, MUTED).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(tlab))

        def partial(K):
            s = V[:, :K] @ a[:K]
            return VGroup(*[
                Rectangle(width=8 / n * 0.62, height=max(abs(v) / ymax * 1.6, 1e-3), stroke_width=0,
                          fill_color=STOCK if v > 0 else POS, fill_opacity=0.85)
                .move_to(center + RIGHT * (-4 + (i + 0.5) * 8 / n) + UP * v / ymax * 0.8)
                for i, v in enumerate(s)])

        cur = partial(1)
        klab = T("modes used: 1   (mode 0 alone)", 22, FG).move_to(center + UP * 1.9 + RIGHT * 2.5)
        self.play(FadeIn(cur), FadeIn(klab))
        share = a[0] ** 2 / (a ** 2).sum()
        pct = T(f"mode 0 alone explains {share:.0%} of that day's cross-section", 20, STOCK)
        pct.next_to(klab, DOWN, buff=0.1)
        self.play(FadeIn(pct))
        self.wait(1.2)
        for K in (2, 3, 4, 6, 9, 12):
            nk = partial(K)
            kl = T(f"modes used: {K}", 22, FG).move_to(klab)
            self.play(Transform(cur, nk), Transform(klab, kl), run_time=0.8)
        done = T("all 12 modes: exact", 22, BOND).move_to(pct)
        self.play(ReplacementTransform(pct, done))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


class S8_Spectrum3D(ThreeDScene):
    """Rolling spectrum of 417 stocks: time × rank × log10 λ."""

    def construct(self):
        self.camera.background_color = BG
        S = load_json("spectrum.json")
        t = np.array(S["t"])
        L = np.log10(np.array(S["lam"]))  # (n_t, K)
        K = L.shape[1]
        lp = np.log10(S["mp_upper"])
        ax = ThreeDAxes(x_range=[2011, 2025, 2], y_range=[0, K, 10], z_range=[-0.2, 2.3, 0.5],
                        x_length=9, y_length=6, z_length=3.0, axis_config={"color": GRID}, tips=False).shift(IN * 0.8)
        self.set_camera_orientation(phi=68 * DEGREES, theta=-60 * DEGREES, zoom=0.6)
        title = T("The market's spectrum through time — 417 stocks", 32, weight="BOLD").to_edge(UP, buff=0.3)
        sub = T("x: time   ·   y: mode rank k   ·   z: log₁₀ λ_k (rolling 252-day correlation)", 20, MUTED)
        sub.next_to(title, DOWN, buff=0.1)
        self.add_fixed_in_frame_mobjects(title, sub)
        self.play(FadeIn(title), FadeIn(sub), Create(ax))

        def f(u, v):
            i = np.interp(u, t, np.arange(len(t)))
            i0 = int(np.clip(np.floor(i), 0, len(t) - 2))
            w = i - i0
            k = np.clip(v, 0, K - 1)
            k0 = int(np.clip(np.floor(k), 0, K - 2))
            wk = k - k0
            z = ((1 - w) * ((1 - wk) * L[i0, k0] + wk * L[i0, k0 + 1])
                 + w * ((1 - wk) * L[i0 + 1, k0] + wk * L[i0 + 1, k0 + 1]))
            return ax.c2p(u, v, z)

        surf = Surface(f, u_range=[t[0], t[-1]], v_range=[0, K - 1], resolution=(56, 30),
                       stroke_width=0.3, stroke_color=BG, fill_opacity=0.92, checkerboard_colors=False)
        surf.set_fill_by_value(axes=ax, colorscale=[("#1e3a8a", -0.2), (CYAN, lp), (GOLD, 1.0), (POS, 2.2)], axis=2)
        plane = Surface(lambda u, v: ax.c2p(u, v, lp), u_range=[t[0], t[-1]], v_range=[0, K - 1],
                        resolution=(2, 2), fill_color=CYAN, fill_opacity=0.25, stroke_width=0,
                        checkerboard_colors=False)
        self.play(Create(surf), run_time=4)
        lbl = T("Marchenko–Pastur noise floor", 22, CYAN).move_to(ax.c2p(2018, K + 6, lp + 0.15))
        self.add_fixed_orientation_mobjects(lbl)
        lbl.set_opacity(0)
        self.play(FadeIn(plane), lbl.animate.set_opacity(1), run_time=1.5)
        xl = VGroup(*[T(str(y), 18, MUTED).move_to(ax.c2p(y, -5, -0.2))
                      for y in (2012, 2016, 2020, 2024)])
        kl = T("mode rank k", 20, MUTED).move_to(ax.c2p(2026, K / 2, -0.2))
        self.add_fixed_orientation_mobjects(*xl, kl)
        cap = T("A tall ridge (the market mode, peaking in 2020) and a handful of sector modes above the floor;\n"
                "everything else sits in the noise bulk.", 20, FG).to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(cap)
        self.play(FadeIn(cap))
        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(7)
        self.stop_ambient_camera_rotation()
        self.move_camera(phi=85 * DEGREES, theta=-90 * DEGREES, run_time=3)
        cap2 = T(f"Only {min(S['n_signal'])}–{max(S['n_signal'])} of 417 modes ever clear the floor.", 22, FG).to_edge(DOWN, buff=0.3)
        self.remove_fixed_in_frame_mobjects(cap)
        self.remove(cap)
        self.add_fixed_in_frame_mobjects(cap2)
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


class S9_Findings(Scene):
    def construct(self):
        self.camera.background_color = BG
        F = load_json("findings.json")["buckets"]
        title = T("From oscillators to mean reversion", 38, weight="BOLD").to_edge(UP, buff=0.3)
        self.play(FadeIn(title))
        lines = VGroup(
            T("Overdamped oscillators driven by noise (an Ornstein–Uhlenbeck market):", 22, FG),
            MathTex(r"dx=-Kx\,dt+s\,dW\quad\Longrightarrow\quad \Sigma=\tfrac{s^2}{2}K^{-1},"
                    r"\qquad \lambda_k=\frac{s^2}{2\,\omega_k^2}", font_size=36),
            T("big λ  =  soft spring  =  slow, trending          small λ  =  stiff spring  =  fast reversion", 21, GOLD),
        ).arrange(DOWN, buff=0.3).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(lines[0]), run_time=0.8)
        self.play(Write(lines[1]), run_time=2)
        self.play(FadeIn(lines[2]))
        self.wait(2)
        self.play(lines.animate.scale(0.75).next_to(title, DOWN, buff=0.15))

        labels = ["Market\n(k=0)", "Signal\n(above MP)", "Edge of\nbulk", "Upper\nbulk", "Middle\nbulk", "Lower\nbulk"]
        ax = Axes(x_range=[0, 6, 1], y_range=[-0.2, 1.0, 0.2], x_length=9, y_length=3.0,
                  axis_config={"color": GRID}, tips=False).to_edge(DOWN, buff=1.45)
        zero = Line(ax.c2p(0, 0), ax.c2p(6, 0), color=MUTED)
        yl = VGroup(*[T(f"{v:.1f}", 14, MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0, 0.4, 0.8)])
        ylab = T("out-of-sample Sharpe, 2013–2024", 16, MUTED).rotate(PI / 2).next_to(yl, LEFT, buff=0.15)
        xl = VGroup(*[T(s, 15, MUTED).next_to(ax.c2p(i + 0.5, -0.2), DOWN, buff=0.08) for i, s in enumerate(labels)])
        self.play(Create(ax), Create(zero), FadeIn(yl), FadeIn(ylab), FadeIn(xl))
        bars = VGroup()
        for i, b in enumerate(F):
            v = b["sharpe"]
            h = abs(v) / 1.2 * ax.y_length
            r = Rectangle(width=0.9, height=max(h, 1e-3), fill_opacity=0.9, stroke_width=0,
                          fill_color=BOND if b["t"] > 1.96 else (GOLD if v > 0.3 else MUTED))
            r.move_to(ax.c2p(i + 0.5, 0), aligned_edge=DOWN if v >= 0 else UP)
            tl = T(f"t = {b['t']:.1f}", 14, FG).next_to(r, UP if v >= 0 else DOWN, buff=0.06)
            bars.add(VGroup(r, tl))
        arrow = Arrow(ax.c2p(0.3, 0.95), ax.c2p(5.7, 0.95), color=GOLD, buff=0, stroke_width=3)
        al = T("smaller eigenvalue  →", 16, GOLD).next_to(arrow, UP, buff=0.05)
        self.play(LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in bars], lag_ratio=0.25),
                  LaggedStart(*[FadeIn(b[1]) for b in bars], lag_ratio=0.25), run_time=3)
        self.play(GrowArrow(arrow), FadeIn(al))
        cap = T("The 'signal' modes do not revert. Reversion lives deep in the low-variance bulk —\n"
                "the direction the physics predicts, though the effect is small and costly to trade.",
                20, FG).to_edge(DOWN, buff=0.15)
        self.play(FadeIn(cap))
        self.wait(4)
        end = T("Normal Modes of the Market  ·  github.com/besufikad-a/normal-modes-of-the-market", 20, MUTED)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)
        self.play(FadeIn(end))
        self.wait(2)
        self.play(FadeOut(end))
