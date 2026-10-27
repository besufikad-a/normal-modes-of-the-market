"""Scenes 1–3: fifty years of stocks, bonds and gold.

S1_History            value (log) and rolling correlation vs. time, drawn live
S2_Relationship3D     3-D cloud: stock return × bond return × time
S3_CorrelationRibbons 3-D ribbons: time × asset pair × correlation
"""
from __future__ import annotations

import numpy as np
from manim import *

from common import (BG, BOND, FG, GOLD, GRID, MUTED, NEG, ORANGE, POS, PURPLE, STOCK, CYAN,
                    T, corr_color, load_history)

H = load_history()
YR = H["year"].values
T0, T1 = 1975.0, 2027.0

EVENTS = [
    (1980.0, "Volcker\nrates 15%"),
    (1987.8, "Black\nMonday"),
    (2000.2, "Dot-com\npeak"),
    (2008.7, "Global Financial\nCrisis"),
    (2020.2, "COVID"),
    (2022.4, "Inflation\nshock"),
]


class S1_History(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = T("Fifty years of three assets", 40, weight="BOLD").to_edge(UP, buff=0.35)
        sub = T("S&P 500 (price), 10-year Treasury (total return), gold  ·  1975 – 2026", 22, MUTED)
        sub.next_to(title, DOWN, buff=0.12)
        self.play(FadeIn(title, shift=DOWN * 0.2), FadeIn(sub), run_time=1.2)

        # ---------------- top axes: value of $1, log scale
        top = Axes(
            x_range=[T0, T1, 5], y_range=[-0.3, 2.3, 1], x_length=10.8, y_length=2.7,
            axis_config={"color": GRID, "stroke_width": 2, "include_ticks": True},
            x_axis_config={"include_numbers": False}, tips=False,
        ).shift(UP * 0.75 + RIGHT * 0.35)
        ylab = VGroup(*[
            T(s, 18, MUTED).next_to(top.c2p(T0, v), LEFT, buff=0.12)
            for v, s in ((0, "$1"), (1, "$10"), (2, "$100"))
        ])
        grid = VGroup(*[
            DashedLine(top.c2p(T0, v), top.c2p(T1, v), color=GRID, stroke_width=1, dash_length=0.06)
            for v in (0, 1, 2)
        ])
        ytitle = T("Growth of $1  (log scale)", 18, MUTED).rotate(PI / 2).next_to(ylab, LEFT, buff=0.15)

        # ---------------- bottom axes: 36-month rolling correlation
        bot = Axes(
            x_range=[T0, T1, 5], y_range=[-1, 1, 0.5], x_length=10.8, y_length=1.8,
            axis_config={"color": GRID, "stroke_width": 2}, tips=False,
        ).next_to(top, DOWN, buff=0.4).align_to(top, LEFT)
        zero = DashedLine(bot.c2p(T0, 0), bot.c2p(T1, 0), color=MUTED, stroke_width=1.2)
        blab = VGroup(*[T(f"{v:+.0f}" if v else "0", 16, MUTED).next_to(bot.c2p(T0, v), LEFT, buff=0.12)
                        for v in (-1, 0, 1)])
        btitle = T("Correlation (36-mo)", 16, MUTED).rotate(PI / 2).next_to(blab, LEFT, buff=0.15)
        xlab = VGroup(*[T(str(y), 16, MUTED).next_to(bot.c2p(y, -1), DOWN, buff=0.12)
                        for y in range(1975, 2030, 5)])

        self.play(Create(top), Create(bot), FadeIn(ylab), FadeIn(grid), FadeIn(ytitle),
                  FadeIn(blab), FadeIn(btitle), Create(zero), FadeIn(xlab), run_time=1.5)

        legend = VGroup(
            *[VGroup(Line(ORIGIN, RIGHT * 0.4, color=c, stroke_width=5), T(n, 18, c)).arrange(RIGHT, buff=0.1)
              for c, n in ((STOCK, "Stocks"), (BOND, "Bonds"), (GOLD, "Gold"))]
        ).arrange(RIGHT, buff=0.35).next_to(top, UP, buff=0.05).align_to(top, LEFT).shift(RIGHT * 0.3)
        legend2 = VGroup(
            *[VGroup(Line(ORIGIN, RIGHT * 0.4, color=c, stroke_width=5), T(n, 16, c)).arrange(RIGHT, buff=0.1)
              for c, n in ((PURPLE, "stocks ↔ bonds"), (ORANGE, "stocks ↔ gold"))]
        ).arrange(RIGHT, buff=0.35).next_to(bot, UP, buff=0.02).align_to(bot, RIGHT)
        self.play(FadeIn(legend), FadeIn(legend2), run_time=0.6)

        tracker = ValueTracker(T0 + 0.1)

        def series_line(ax, col, color, width=3, log=True):
            def build():
                t = tracker.get_value()
                m = (YR <= t) & np.isfinite(H[col].values)
                ys = H[col].values[m]
                if log:
                    ys = np.log10(ys)
                if m.sum() < 2:
                    return VMobject()
                pts = [ax.c2p(x, y) for x, y in zip(YR[m], ys)]
                v = VMobject(stroke_color=color, stroke_width=width)
                v.set_points_as_corners(pts)
                return v
            return always_redraw(build)

        lines = [series_line(top, "v_stock", STOCK), series_line(top, "v_bond", BOND),
                 series_line(top, "v_gold", GOLD),
                 series_line(bot, "c_stock_bond", PURPLE, 3, log=False),
                 series_line(bot, "c_stock_gold", ORANGE, 2.5, log=False)]

        cursor = always_redraw(lambda: DashedLine(
            top.c2p(tracker.get_value(), 2.3), bot.c2p(tracker.get_value(), -1),
            color=FG, stroke_width=1, stroke_opacity=0.5, dash_length=0.05))
        year_lbl = always_redraw(lambda: T(f"{int(tracker.get_value())}", 30, FG, "BOLD")
                                 .next_to(top, UP, buff=0.0).align_to(top, RIGHT))
        self.add(*lines, cursor, year_lbl)

        # Draw history in segments, pausing at events.
        shown = VGroup()
        prev = T0 + 0.1
        for (ty, name) in EVENTS + [(2026.75, None)]:
            dur = max(0.6, (ty - prev) / 52 * 16)
            self.play(tracker.animate.set_value(ty), run_time=dur, rate_func=linear)
            prev = ty
            if name:
                mk = VGroup(
                    DashedLine(top.c2p(ty, -0.3), top.c2p(ty, 2.3), color=MUTED, stroke_width=1,
                               dash_length=0.05, stroke_opacity=0.7),
                    T(name, 13, MUTED).move_to(top.c2p(ty, 1.75 if ty < 2000 else (-0.05 if name == "COVID" else 0.35 if ty > 2021 else -0.05))),
                )
                shown.add(mk)
                self.play(FadeIn(mk), run_time=0.35)
        self.wait(0.5)

        # Freeze the redrawn lines.
        for ln in lines:
            ln.clear_updaters()
        cursor.clear_updaters()
        self.play(FadeOut(cursor), FadeOut(year_lbl), run_time=0.4)

        # Highlight the negative stock–bond era.
        neg = Rectangle(
            width=bot.c2p(2021.5, 0)[0] - bot.c2p(2000.0, 0)[0], height=bot.y_length,
            fill_color=NEG, fill_opacity=0.15, stroke_width=0,
        ).move_to(bot.c2p((2000 + 2021.5) / 2, 0))
        note = T("2000 – 2021: bonds hedged stocks (negative correlation)", 20, NEG)
        note.to_edge(DOWN, buff=0.12)
        self.play(FadeIn(neg), Write(note), run_time=1.5)
        self.wait(1.2)
        note2 = T("1975 – 1999 and 2022 –: they moved together (positive correlation)", 20, POS)
        note2.move_to(note)
        self.play(ReplacementTransform(note, note2), run_time=1)
        self.wait(1.2)
        q = T("Correlation is not a constant — it is a state of the system.", 24, FG, "BOLD")
        q.move_to(note2)
        self.play(ReplacementTransform(note2, q), run_time=1)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


class S2_Relationship3D(ThreeDScene):
    """Each month is a point (stock return, bond return, time)."""

    def construct(self):
        self.camera.background_color = BG
        d = H.dropna(subset=["c_stock_bond"])
        ax = ThreeDAxes(
            x_range=[-0.15, 0.15, 0.05], y_range=[-0.10, 0.10, 0.05], z_range=[1977, 2027, 10],
            x_length=6, y_length=6, z_length=4.6,
            axis_config={"color": GRID, "stroke_width": 2}, tips=False,
        ).shift(IN * 1.2)
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES, zoom=0.9)

        title = T("The stock–bond relationship, month by month", 34, weight="BOLD").to_edge(UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(title)
        self.play(FadeIn(title), Create(ax), run_time=1.5)

        xl = T("stock return", 22, STOCK).next_to(ax.c2p(0.15, 0, 1977), RIGHT, buff=0.1)
        yl = T("bond return", 22, BOND).next_to(ax.c2p(0, 0.10, 1977), UP, buff=0.1)
        self.add_fixed_orientation_mobjects(xl, yl)

        # Flat dots that always face the camera: far cheaper than spheres.
        dots = VGroup(*[
            Dot(ax.c2p(np.clip(r.r_stock, -0.15, 0.15), np.clip(r.r_bond, -0.1, 0.1), r.year),
                radius=0.04, color=corr_color(r.c_stock_bond))
            for r in d.itertuples()
        ])
        self.add_fixed_orientation_mobjects(*dots)
        for p in dots:
            p.set_opacity(0)
        # Top-down view first: just the scatter, time collapsed.
        cap = T("Looking down the time axis: a cloud with no clear tilt", 22, MUTED).to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(cap)
        self.play(LaggedStart(*[p.animate.set_opacity(1) for p in dots], lag_ratio=0.004), FadeIn(cap), run_time=3)
        self.wait(0.8)

        # Decade principal axes (eigenvector of the 2x2 correlation of that decade).
        axes_lines = VGroup()
        dec_labels = []
        for dec in range(1975, 2026, 10):
            g = d[(d.year >= dec) & (d.year < dec + 10)]
            if len(g) < 24:
                continue
            rho = np.corrcoef(g.r_stock, g.r_bond)[0, 1]
            Cv = np.cov(g.r_stock, g.r_bond)
            w, V = np.linalg.eigh(Cv)
            v = V[:, -1] * np.sqrt(w[-1]) * 2.2
            zc = dec + 5
            seg = Line3D(ax.c2p(-v[0], -v[1], zc), ax.c2p(v[0], v[1], zc),
                         color=corr_color(rho * 1.8), thickness=0.03)
            lbl = T(f"{dec}s  ρ={rho:+.2f}", 18, corr_color(rho * 1.8))
            lbl.move_to(ax.c2p(0.16, 0.10, zc))
            axes_lines.add(seg)
            dec_labels.append(lbl)

        cap2 = T("Tilt the cloud: time runs up the vertical axis", 22, MUTED).to_edge(DOWN, buff=0.4)
        self.remove_fixed_in_frame_mobjects(cap)
        self.add_fixed_in_frame_mobjects(cap2)
        self.remove(cap)
        tl = T("time ↑", 22, FG).move_to(ax.c2p(0, -0.11, 2030)).set_opacity(0)
        self.add_fixed_orientation_mobjects(tl)
        self.move_camera(phi=72 * DEGREES, theta=-50 * DEGREES, zoom=0.62, run_time=4,
                         added_anims=[tl.animate.set_opacity(1)])
        self.wait(0.5)
        for lb in dec_labels:
            lb.set_opacity(0)
        self.add_fixed_orientation_mobjects(*dec_labels)
        self.play(LaggedStart(*[Create(m) for m in axes_lines], lag_ratio=0.3),
                  LaggedStart(*[lb.animate.set_opacity(1) for lb in dec_labels], lag_ratio=0.3), run_time=3)
        cap3 = T("Each decade's principal axis: the relationship rotates through time",
                 22, FG).to_edge(DOWN, buff=0.4)
        self.remove_fixed_in_frame_mobjects(cap2)
        self.remove(cap2)
        self.add_fixed_in_frame_mobjects(cap3)
        self.begin_ambient_camera_rotation(rate=0.18)
        self.wait(6)
        self.stop_ambient_camera_rotation()
        legend = VGroup(
            VGroup(Dot(color=POS), T("positive correlation", 18, POS)).arrange(RIGHT, buff=0.1),
            VGroup(Dot(color=NEG), T("negative correlation", 18, NEG)).arrange(RIGHT, buff=0.1),
        ).arrange(DOWN, aligned_edge=LEFT).to_corner(UR).shift(DOWN * 0.8)
        self.add_fixed_in_frame_mobjects(legend)
        self.play(FadeIn(legend))
        self.move_camera(phi=78 * DEGREES, theta=-25 * DEGREES, zoom=0.62, run_time=3)
        cap4 = T("Red until ~2000, blue until 2021, red again since 2022: the sign of the relationship flips",
                 22, FG).to_edge(DOWN, buff=0.4)
        self.remove_fixed_in_frame_mobjects(cap3)
        self.remove(cap3)
        self.add_fixed_in_frame_mobjects(cap4)
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


class S3_CorrelationRibbons(ThreeDScene):
    """Time × asset pair × correlation, as three ribbons."""

    def construct(self):
        self.camera.background_color = BG
        d = H.dropna(subset=["c_stock_bond", "c_stock_gold", "c_bond_gold"])
        ax = ThreeDAxes(
            x_range=[1977, 2027, 10], y_range=[0, 3, 1], z_range=[-1, 1, 0.5],
            x_length=9, y_length=4.5, z_length=3.5,
            axis_config={"color": GRID, "stroke_width": 2}, tips=False,
        )
        self.set_camera_orientation(phi=65 * DEGREES, theta=-70 * DEGREES, zoom=0.8)
        title = T("All three relationships at once", 34, weight="BOLD").to_edge(UP, buff=0.3)
        sub = T("x: time   ·   y: asset pair   ·   z: 36-month correlation", 22, MUTED).next_to(title, DOWN, buff=0.1)
        self.add_fixed_in_frame_mobjects(title, sub)
        self.play(FadeIn(title), FadeIn(sub), Create(ax), run_time=1.5)

        zero = Surface(lambda u, v: ax.c2p(u, v, 0), u_range=[1977, 2027], v_range=[0, 3],
                       resolution=(2, 2), fill_opacity=0.08, fill_color=FG, stroke_width=0.5,
                       stroke_color=GRID, checkerboard_colors=False)
        self.add(zero)

        pairs = [("c_stock_bond", "stocks ↔ bonds", 0.5, PURPLE),
                 ("c_stock_gold", "stocks ↔ gold", 1.5, ORANGE),
                 ("c_bond_gold", "bonds ↔ gold", 2.5, CYAN)]
        ribbons = VGroup()
        labels = VGroup()
        step = 3
        for col, name, yy, color in pairs:
            yrs = d.year.values[::step]
            cs = d[col].values[::step]
            strips = VGroup()
            for i in range(len(yrs) - 1):
                p = [ax.c2p(yrs[i], yy - 0.3, cs[i]), ax.c2p(yrs[i + 1], yy - 0.3, cs[i + 1]),
                     ax.c2p(yrs[i + 1], yy + 0.3, cs[i + 1]), ax.c2p(yrs[i], yy + 0.3, cs[i])]
                poly = Polygon(*p, stroke_width=0, fill_opacity=0.9,
                               fill_color=corr_color((cs[i] + cs[i + 1]) / 2 * 1.6))
                strips.add(poly)
            ribbons.add(strips)
            lbl = T(name, 22, color)
            lbl.move_to(ax.c2p(1971, yy, 0.9 - 0.4 * len(labels)))
            labels.add(lbl)

        for strips, lbl in zip(ribbons, labels):
            lbl.set_opacity(0)
            self.add_fixed_orientation_mobjects(lbl)
            self.play(Create(strips, lag_ratio=0.02), lbl.animate.set_opacity(1), run_time=2.5)
        yl = VGroup(*[T(f"{v:+.0f}" if v else "0", 18, MUTED)
                      .move_to(ax.c2p(1977, 3.2, v)) for v in (-1, 0, 1)])
        xl = VGroup(*[T(str(y), 18, MUTED).move_to(ax.c2p(y, -0.35, -1)) for y in (1980, 1990, 2000, 2010, 2020)])
        self.add_fixed_orientation_mobjects(*yl, *xl)
        self.play(FadeIn(yl), FadeIn(xl))
        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(5)
        self.stop_ambient_camera_rotation()
        self.move_camera(phi=80 * DEGREES, theta=-82 * DEGREES, run_time=3)
        cap = T("Red = moving together, blue = hedging each other.\nGold has stayed close to zero with both.",
                22, FG).to_edge(DOWN, buff=0.35)
        self.add_fixed_in_frame_mobjects(cap)
        self.play(FadeIn(cap))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)
