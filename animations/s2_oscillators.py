"""Scenes 4–6: coupled oscillators and their normal modes.

S4_TwoOscillators   N = 2: stocks and bonds as two coupled masses; each mode
                    separately, then their combination; link to λ = 1 ± ρ.
S5_ChainModes       N = 3, 5, 8 beads on a string: every mode separately,
                    then the superposition with its modal amplitudes.
S6_ModeSurface3D    N = 16 in 3-D: bead × mode × displacement, all modes
                    oscillating at once, and their sum.
"""
from __future__ import annotations

import numpy as np
from manim import *

from common import (BG, BOND, FG, GOLD, GRID, MODE_COLORS, MUTED, NEG, POS, PURPLE, STOCK,
                    T, chain_modes, load_history, spring)

# ----------------------------------------------------------------- N = 2


class S4_TwoOscillators(Scene):
    k, kc = 1.0, 0.6

    def construct(self):
        self.camera.background_color = BG
        w1, w2 = np.sqrt(self.k), np.sqrt(self.k + 2 * self.kc)

        title = T("Two coupled oscillators  (N = 2)", 38, weight="BOLD").to_edge(UP, buff=0.3)
        self.play(FadeIn(title, shift=DOWN * 0.2))

        # ---------------- the mechanical system
        y0 = 1.65
        wallL = Line(UP * 0.55, DOWN * 0.55, color=GRID, stroke_width=6).move_to([-5.2, y0, 0])
        wallR = wallL.copy().move_to([5.2, y0, 0])
        hatch = VGroup(*[Line([-5.2, y0 + d, 0], [-5.45, y0 + d - 0.2, 0], color=GRID, stroke_width=2)
                         for d in np.linspace(-0.45, 0.55, 6)],
                       *[Line([5.2, y0 + d, 0], [5.45, y0 + d + 0.2, 0], color=GRID, stroke_width=2)
                         for d in np.linspace(-0.55, 0.45, 6)])
        x1t, x2t = ValueTracker(0.0), ValueTracker(0.0)
        eq1, eq2, sc = -1.7, 1.7, 0.9

        def mass(tr, eq, color, name):
            def build():
                sq = Square(0.85, fill_color=color, fill_opacity=0.9, stroke_color=FG, stroke_width=1.5)
                sq.move_to([eq + sc * tr.get_value(), y0, 0])
                return VGroup(sq, T(name, 18, BG, "BOLD").move_to(sq))
            return always_redraw(build)

        m1 = mass(x1t, eq1, STOCK, "Stocks")
        m2 = mass(x2t, eq2, BOND, "Bonds")
        sp = always_redraw(lambda: VGroup(
            spring([-5.2, y0, 0], [eq1 + sc * x1t.get_value() - 0.43, y0, 0]),
            spring([eq1 + sc * x1t.get_value() + 0.43, y0, 0], [eq2 + sc * x2t.get_value() - 0.43, y0, 0],
                   color=PURPLE, coils=9),
            spring([eq2 + sc * x2t.get_value() + 0.43, y0, 0], [5.2, y0, 0]),
        ))
        klab = VGroup(T("k", 22, MUTED).move_to([-3.9, y0 + 0.5, 0]),
                      T("coupling k꜀", 22, PURPLE).move_to([0, y0 + 0.55, 0]),
                      T("k", 22, MUTED).move_to([3.9, y0 + 0.5, 0]))
        self.play(Create(wallL), Create(wallR), FadeIn(hatch), FadeIn(sp), FadeIn(m1), FadeIn(m2),
                  FadeIn(klab), run_time=1.5)

        eqn = MathTex(r"\ddot{x} = -Kx,\quad K=\begin{pmatrix}k+k_c & -k_c\\ -k_c & k+k_c\end{pmatrix}",
                      font_size=34, color=FG).move_to([0, 0.35, 0])
        self.play(Write(eqn), run_time=1.5)
        self.wait(0.5)

        # ---------------- trace axes
        ax = Axes(x_range=[0, 20, 5], y_range=[-1.3, 1.3, 1], x_length=10.5, y_length=2.6,
                  axis_config={"color": GRID}, tips=False).move_to([0.3, -2.2, 0])
        axl = VGroup(T("displacement", 16, MUTED).rotate(PI / 2).next_to(ax, LEFT, buff=0.15),
                     T("time →", 16, MUTED).next_to(ax, DOWN, buff=0.1).align_to(ax, RIGHT))
        tt = ValueTracker(0.0)

        def trace(fn, color, width=3.5, axes=None):
            axes = axes or ax

            def build():
                t = tt.get_value()
                if t < 0.05:
                    return VMobject()
                s = np.linspace(0, t, max(2, int(t * 12)))
                v = VMobject(stroke_color=color, stroke_width=width)
                v.set_points_smoothly([axes.c2p(si, fn(si)) for si in s])
                return v
            return always_redraw(build)

        def run(f1, f2, label, tmax=20, rt=8, extra=()):
            x1t.add_updater(lambda m: m.set_value(f1(tt.get_value())))
            x2t.add_updater(lambda m: m.set_value(f2(tt.get_value())))
            tr1, tr2 = trace(f1, STOCK), trace(f2, BOND, 2.5)
            self.add(tr1, tr2, x1t, x2t)
            self.play(FadeIn(label), run_time=0.6)
            self.play(tt.animate.set_value(tmax), run_time=rt, rate_func=linear)
            x1t.clear_updaters()
            x2t.clear_updaters()
            for m in (tr1, tr2):
                m.clear_updaters()
            return VGroup(tr1, tr2)

        self.play(FadeOut(eqn), Create(ax), FadeIn(axl), run_time=1)

        # Mode 1: in phase.
        lab1 = VGroup(T("Mode 1 — in phase", 26, STOCK, "BOLD"),
                      MathTex(r"v_1=\tfrac{1}{\sqrt2}(1,\,1),\ \ \omega_1=\sqrt{k}", font_size=30)
                      ).arrange(RIGHT, buff=0.4).move_to([0, 0.45, 0])
        A = 1.0
        tr = run(lambda t: A * np.cos(w1 * t), lambda t: A * np.cos(w1 * t), lab1)
        note1 = T("both move together: the coupling spring never stretches  →  the 'market' mode",
                  20, MUTED).move_to([0, -0.55, 0])
        self.play(FadeIn(note1))
        self.wait(1.2)
        self.play(FadeOut(tr), FadeOut(lab1), FadeOut(note1))
        tt.set_value(0)

        # Mode 2: out of phase.
        lab2 = VGroup(T("Mode 2 — out of phase", 26, POS, "BOLD"),
                      MathTex(r"v_2=\tfrac{1}{\sqrt2}(1,\,-1),\ \ \omega_2=\sqrt{k+2k_c}", font_size=30)
                      ).arrange(RIGHT, buff=0.4).move_to([0, 0.45, 0])
        tr = run(lambda t: A * np.cos(w2 * t), lambda t: -A * np.cos(w2 * t), lab2)
        note2 = T("they move against each other: the coupling stretches, so it is stiffer and faster  →  the 'hedge' mode",
                  19, MUTED).move_to([0, -0.55, 0])
        self.play(FadeIn(note2))
        self.wait(1.2)
        self.play(FadeOut(tr), FadeOut(lab2), FadeOut(note2), FadeOut(ax), FadeOut(axl))
        tt.set_value(0)

        # Combination: start with only stocks displaced -> beats.
        lab3 = VGroup(T("Combination", 26, GOLD, "BOLD"),
                      MathTex(r"x(t)=\tfrac{1}{\sqrt2}\left(q_1\cos\omega_1 t\, v_1+q_2\cos\omega_2 t\, v_2\right)",
                              font_size=30)).arrange(RIGHT, buff=0.4).move_to([0, 0.5, 0])
        axq1 = Axes(x_range=[0, 30, 5], y_range=[-1.1, 1.1, 1], x_length=5.0, y_length=1.15,
                    axis_config={"color": GRID}, tips=False).move_to([-3.4, -1.2, 0])
        axq2 = axq1.copy().move_to([-3.4, -2.8, 0])
        axx = Axes(x_range=[0, 30, 5], y_range=[-1.3, 1.3, 1], x_length=5.6, y_length=2.8,
                   axis_config={"color": GRID}, tips=False).move_to([3.3, -2.0, 0])
        labs = VGroup(T("mode 1 coordinate", 16, STOCK).next_to(axq1, UP, buff=0.02),
                      T("mode 2 coordinate", 16, POS).next_to(axq2, UP, buff=0.02),
                      T("sum: stocks (blue) and bonds (green)", 16, FG).next_to(axx, UP, buff=0.02),
                      MathTex("+", font_size=40).move_to([-3.4, -2.0, 0]).shift(RIGHT * 2.85),
                      MathTex("=", font_size=40).move_to([0.25, -2.0, 0]))
        self.play(FadeIn(lab3), Create(axq1), Create(axq2), Create(axx), FadeIn(labs), run_time=1.5)
        a = 1 / np.sqrt(2)
        q1 = lambda t: a * np.cos(w1 * t)
        q2 = lambda t: a * np.cos(w2 * t)
        # Clean definitions: x1 = (cos w1 t + cos w2 t)/2, x2 = (cos w1 t - cos w2 t)/2
        f1 = lambda t: 0.5 * (np.cos(w1 * t) + np.cos(w2 * t))
        f2 = lambda t: 0.5 * (np.cos(w1 * t) - np.cos(w2 * t))
        x1t.add_updater(lambda m: m.set_value(f1(tt.get_value())))
        x2t.add_updater(lambda m: m.set_value(f2(tt.get_value())))
        trs = VGroup(trace(q1, STOCK, 3, axq1), trace(q2, POS, 3, axq2),
                     trace(f1, STOCK, 3, axx), trace(f2, BOND, 2.5, axx))
        self.add(trs, x1t, x2t)
        self.play(tt.animate.set_value(30), run_time=12, rate_func=linear)
        x1t.clear_updaters()
        x2t.clear_updaters()
        for m in trs:
            m.clear_updaters()
        note3 = T("Push only the stocks mass: energy sloshes between the two — 'beats' from two frequencies",
                  19, MUTED).to_edge(DOWN, buff=0.12)
        self.play(FadeIn(note3))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in (trs, axq1, axq2, axx, labs, lab3, note3)])

        # ---------------- the market link: lambda = 1 ± rho
        self.market_link(title)

    def market_link(self, title):
        H = load_history().dropna(subset=["c_stock_bond"])
        box = VGroup(
            T("For two assets the correlation matrix always has the same two modes:", 22, FG),
            MathTex(r"C=\begin{pmatrix}1&\rho\\ \rho&1\end{pmatrix},\quad "
                    r"v_1=\tfrac{1}{\sqrt2}(1,1)\ \ \lambda_1=1+\rho,\quad "
                    r"v_2=\tfrac{1}{\sqrt2}(1,-1)\ \ \lambda_2=1-\rho", font_size=32),
        ).arrange(DOWN, buff=0.25).move_to([0, 0.15, 0])
        self.play(FadeIn(box[0]), Write(box[1]), run_time=2)
        self.wait(1)

        ax = Axes(x_range=[0, 3, 1], y_range=[0, 2, 0.5], x_length=4.2, y_length=2.2,
                  axis_config={"color": GRID}, tips=False).move_to([-2.4, -2.15, 0])
        ylab = VGroup(*[T(f"{v:g}", 14, MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0, 1, 2)])
        tr = ValueTracker(H.year.iloc[0])

        def rho_now():
            return float(np.interp(tr.get_value(), H.year.values, H.c_stock_bond.values))

        bars = always_redraw(lambda: VGroup(
            Rectangle(width=0.9, height=max(1e-3, ax.y_length * (1 + rho_now()) / 2), fill_color=STOCK,
                      fill_opacity=0.85, stroke_width=0).next_to(ax.c2p(1, 0), UP, buff=0),
            Rectangle(width=0.9, height=max(1e-3, ax.y_length * (1 - rho_now()) / 2), fill_color=POS,
                      fill_opacity=0.85, stroke_width=0).next_to(ax.c2p(2, 0), UP, buff=0),
        ))
        blab = VGroup(MathTex(r"\lambda_1", font_size=30, color=STOCK).next_to(ax.c2p(1, 0), DOWN),
                      MathTex(r"\lambda_2", font_size=30, color=POS).next_to(ax.c2p(2, 0), DOWN))
        info = always_redraw(lambda: VGroup(
            T(f"{int(tr.get_value())}", 44, FG, "BOLD"),
            MathTex(rf"\rho = {rho_now():+.2f}", font_size=40,
                    color=POS if rho_now() > 0 else NEG),
            T("in-phase (market) mode dominates" if rho_now() > 0 else "out-of-phase (hedge) mode dominates",
              20, STOCK if rho_now() > 0 else POS),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([2.9, -2.1, 0]))
        cap = T("Stocks vs. bonds, 36-month correlation, 1978 – 2026", 18, MUTED).next_to(ax, UP, buff=0.15)
        self.play(Create(ax), FadeIn(ylab), FadeIn(bars), FadeIn(blab), FadeIn(info), FadeIn(cap))
        self.play(tr.animate.set_value(H.year.iloc[-1]), run_time=12, rate_func=linear)
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


# ----------------------------------------------------------------- N = 3, 5, 8


class S5_ChainModes(Scene):
    Ns = (3, 5, 8)

    def construct(self):
        self.camera.background_color = BG
        title = T("N coupled oscillators: beads on a string", 38, weight="BOLD").to_edge(UP, buff=0.3)
        eq = MathTex(r"K=\begin{pmatrix}2&-1&&\\-1&2&-1&\\&\ddots&\ddots&\ddots\\&&-1&2\end{pmatrix},\qquad "
                     r"v_{ik}=\sin\frac{ik\pi}{N+1},\qquad \omega_k=2\sin\frac{k\pi}{2(N+1)}",
                     font_size=32).next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(title), Write(eq), run_time=2)
        self.wait(1.5)
        self.play(FadeOut(eq))
        for N in self.Ns:
            self.one_N(N, title)
        self.play(*[FadeOut(m) for m in self.mobjects])

    def one_N(self, N, title):
        omega, V = chain_modes(N)
        tt = ValueTracker(0.0)
        speed = 3.0
        x0, x1 = -4.6, 1.8
        xs = np.linspace(x0, x1, N + 2)[1:-1]

        head = T(f"N = {N}:  each mode on its own", 28, GOLD, "BOLD").next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(head, shift=DOWN * 0.1))

        top, bottom = 2.0, -3.05
        rows = np.linspace(top, bottom, N) if N > 1 else [0]
        gap = (top - bottom) / max(N - 1, 1)
        amp = min(0.42, 0.42 * gap)

        def row_mob(k, yrow):
            col = MODE_COLORS[k % len(MODE_COLORS)]

            def build():
                c = np.cos(omega[k] * tt.get_value())
                ys = yrow + amp * V[:, k] / np.abs(V[:, k]).max() * c
                pts = [[x0, yrow, 0]] + [[x, y, 0] for x, y in zip(xs, ys)] + [[x1, yrow, 0]]
                string = VMobject(stroke_color=col, stroke_width=2).set_points_as_corners(pts)
                beads = VGroup(*[Dot([x, y, 0], radius=0.09 if N <= 5 else 0.07, color=col) for x, y in zip(xs, ys)])
                return VGroup(string, beads)
            return always_redraw(build)

        mobs = VGroup()
        for k, yr in enumerate(rows):
            base = VGroup(Dot([x0, yr, 0], radius=0.04, color=GRID), Dot([x1, yr, 0], radius=0.04, color=GRID),
                          DashedLine([x0, yr, 0], [x1, yr, 0], color=GRID, stroke_width=1, dash_length=0.05))
            lab = VGroup(T(f"mode {k + 1}", 18 if N <= 5 else 15, MODE_COLORS[k % len(MODE_COLORS)]),
                         MathTex(rf"\omega_{k + 1}={omega[k]:.2f}", font_size=24 if N <= 5 else 20)
                         ).arrange(RIGHT, buff=0.2).move_to([x1 + 1.6, yr, 0])
            mobs.add(VGroup(base, row_mob(k, yr), lab))
        self.play(LaggedStart(*[FadeIn(m) for m in mobs], lag_ratio=0.1), run_time=1.2)
        note = T("higher modes: more sign changes, stiffer, faster", 18, MUTED).to_corner(DR, buff=0.25)
        self.play(FadeIn(note), tt.animate.set_value(8 * speed), run_time=8, rate_func=linear)
        for m in mobs:
            m[1].clear_updaters()
        self.play(FadeOut(mobs), FadeOut(note), FadeOut(head))

        # ---------------- combination
        head2 = T(f"N = {N}:  all modes combined", 28, GOLD, "BOLD").next_to(title, DOWN, buff=0.2)
        rng = np.random.default_rng(N)
        c = rng.normal(0, 1, N) / np.sqrt(np.arange(1, N + 1))
        c /= np.abs(V @ c).max()
        phase = rng.uniform(0, 2 * np.pi, N)
        tt.set_value(0)
        yc = 1.0
        amp2 = 1.0

        def total():
            t = tt.get_value()
            q = c * np.cos(omega * t + phase)
            ys = yc + amp2 * (V @ q)
            pts = [[x0, yc, 0]] + [[x, y, 0] for x, y in zip(xs, ys)] + [[x1, yc, 0]]
            string = VMobject(stroke_color=FG, stroke_width=3).set_points_as_corners(pts)
            beads = VGroup(*[Dot([x, y, 0], radius=0.11, color=GOLD) for x, y in zip(xs, ys)])
            ghosts = VGroup()
            for k in range(N):
                gy = yc + amp2 * V[:, k] * q[k]
                g = VMobject(stroke_color=MODE_COLORS[k % len(MODE_COLORS)], stroke_width=1.5,
                             stroke_opacity=0.55)
                g.set_points_as_corners([[x0, yc, 0]] + [[x, y, 0] for x, y in zip(xs, gy)] + [[x1, yc, 0]])
                ghosts.add(g)
            return VGroup(ghosts, string, beads)

        chain = always_redraw(total)
        base = DashedLine([x0, yc, 0], [x1, yc, 0], color=GRID, stroke_width=1)
        bax = Axes(x_range=[0, N + 1, 1], y_range=[-1, 1, 1], x_length=6.0, y_length=2.2,
                   axis_config={"color": GRID}, tips=False).move_to([-1.4, -2.2, 0])
        bw = 4.0 / (N + 1)

        def bars():
            t = tt.get_value()
            q = c * np.cos(omega * t + phase)
            g = VGroup()
            for k in range(N):
                h = q[k] / np.abs(c).max() * bax.y_length / 2
                r = Rectangle(width=bw, height=max(abs(h), 1e-3), fill_color=MODE_COLORS[k % len(MODE_COLORS)],
                              fill_opacity=0.9, stroke_width=0)
                r.move_to(bax.c2p(k + 1, 0) + UP * h / 2)
                g.add(r)
            return g

        b = always_redraw(bars)
        blabel = VGroup(*[T(str(k + 1), 14, MUTED).next_to(bax.c2p(k + 1, -1), DOWN, buff=0.05) for k in range(N)])
        btitle = T("modal amplitudes  q_k(t) = c_k cos(ω_k t + φ_k)", 18, MUTED).next_to(bax, UP, buff=0.05)
        side = VGroup(T("thin lines: each mode's", 18, MUTED), T("contribution", 18, MUTED),
                      T("thick line: their sum", 18, FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        side.move_to([x1 + 2.3, yc, 0])
        self.play(FadeIn(head2), Create(base), FadeIn(chain), Create(bax), FadeIn(b), FadeIn(blabel),
                  FadeIn(btitle), FadeIn(side), run_time=1.2)
        self.play(tt.animate.set_value(9 * speed), run_time=9, rate_func=linear)
        chain.clear_updaters()
        b.clear_updaters()
        self.play(*[FadeOut(m) for m in (head2, base, chain, bax, b, blabel, btitle, side)])


# ----------------------------------------------------------------- 3-D


class S6_ModeSurface3D(ThreeDScene):
    N = 16

    def construct(self):
        self.camera.background_color = BG
        N = self.N
        omega, V = chain_modes(N)
        ax = ThreeDAxes(x_range=[0, N + 1, 4], y_range=[0, N + 1, 4], z_range=[-1, 1, 0.5],
                        x_length=7, y_length=7, z_length=2.6, axis_config={"color": GRID}, tips=False)
        self.set_camera_orientation(phi=60 * DEGREES, theta=-125 * DEGREES, zoom=0.68)
        title = T(f"All {N} modes at once", 34, weight="BOLD").to_edge(UP, buff=0.3)
        sub = T("x: bead i   ·   y: mode k (slow → fast)   ·   z: displacement  v_ik cos(ω_k t)", 20, MUTED)
        sub.next_to(title, DOWN, buff=0.1)
        self.add_fixed_in_frame_mobjects(title, sub)
        self.play(FadeIn(title), FadeIn(sub), Create(ax))

        xl = T("bead i", 22, FG).move_to(ax.c2p(N / 2, -1.5, -1))
        yl = T("mode k", 22, FG).move_to(ax.c2p(-1.5, N / 2, -1))
        self.add_fixed_orientation_mobjects(xl, yl)
        tt = ValueTracker(0.0)
        # Use exact grid values rather than nearest-index for a smooth sheet.
        def surf_exact():
            t = tt.get_value()

            def f(u, v):
                val = np.sin(u * v * np.pi / (N + 1))
                return ax.c2p(u, v, 0.5 * val * np.cos(2 * np.sin(v * np.pi / (2 * (N + 1))) * t))
            s = Surface(f, u_range=[1, N], v_range=[1, N], resolution=(N - 1, N - 1),
                        stroke_width=0.4, stroke_color=BG, fill_opacity=0.9, checkerboard_colors=False)
            s.set_fill_by_value(axes=ax, colorscale=[(NEG, -0.5), ("#1f2937", 0), (POS, 0.5)], axis=2)
            return s

        sheet = always_redraw(surf_exact)
        self.play(FadeIn(sheet), run_time=1.5)
        self.begin_ambient_camera_rotation(rate=0.1)
        self.play(tt.animate.set_value(30), run_time=10, rate_func=linear)

        # Sum over modes with random amplitudes, drawn as a chain in front.
        rng = np.random.default_rng(7)
        c = rng.normal(0, 1, N) / np.arange(1, N + 1) ** 0.7
        c /= np.abs(V @ c).max()

        def chain():
            t = tt.get_value()
            y = V @ (c * np.cos(omega * t))
            pts = [ax.c2p(0, -1.0, 0)] + [ax.c2p(i + 1, -1.0, y[i] * 0.9) for i in range(N)] + [ax.c2p(N + 1, -1.0, 0)]
            line = VMobject(stroke_color=GOLD, stroke_width=4).set_points_as_corners(pts)
            dots = VGroup(*[Dot(p, radius=0.07, color=GOLD) for p in pts[1:-1]])
            return VGroup(line, dots)

        ch = always_redraw(chain)
        cap = T("Gold chain in front: a general motion = a weighted sum of these rows", 22, FG).to_edge(DOWN, buff=0.35)
        self.add_fixed_in_frame_mobjects(cap)
        self.play(FadeIn(ch), FadeIn(cap))
        self.play(tt.animate.set_value(60), run_time=10, rate_func=linear)
        self.stop_ambient_camera_rotation()
        sheet.clear_updaters()
        ch.clear_updaters()
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)
