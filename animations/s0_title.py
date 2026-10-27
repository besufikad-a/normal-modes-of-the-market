"""Scene 0: title card with a pair of coupled masses."""
from __future__ import annotations

import numpy as np
from manim import *

from common import BG, BOND, FG, MUTED, PURPLE, STOCK, T, spring


class S0_Title(Scene):
    def construct(self):
        self.camera.background_color = BG
        t1 = T("Normal Modes of the Market", 56, weight="BOLD")
        t2 = T("coupled oscillators, correlation, and fifty years of stocks, bonds and gold", 24, MUTED)
        g = VGroup(t1, t2).arrange(DOWN, buff=0.3).shift(UP * 1.2)
        tt = ValueTracker(0)
        w1, w2 = 1.0, 1.48

        def system():
            t = tt.get_value()
            x1 = 0.35 * (np.cos(w1 * t) + np.cos(w2 * t))
            x2 = 0.35 * (np.cos(w1 * t) - np.cos(w2 * t))
            y = -1.6
            a, b = -1.6 + x1, 1.6 + x2
            m1 = Square(0.7, fill_color=STOCK, fill_opacity=0.9, stroke_width=0).move_to([a, y, 0])
            m2 = Square(0.7, fill_color=BOND, fill_opacity=0.9, stroke_width=0).move_to([b, y, 0])
            return VGroup(
                Line([-4.4, y - 0.5, 0], [-4.4, y + 0.5, 0], color=MUTED, stroke_width=5),
                Line([4.4, y - 0.5, 0], [4.4, y + 0.5, 0], color=MUTED, stroke_width=5),
                spring([-4.4, y, 0], [a - 0.35, y, 0]), spring([a + 0.35, y, 0], [b - 0.35, y, 0], color=PURPLE),
                spring([b + 0.35, y, 0], [4.4, y, 0]), m1, m2)

        sys = always_redraw(system)
        self.play(FadeIn(t1, shift=UP * 0.3), run_time=1.2)
        self.play(FadeIn(t2), FadeIn(sys))
        self.play(tt.animate.set_value(14), run_time=6, rate_func=linear)
        sys.clear_updaters()
        self.play(FadeOut(g), FadeOut(sys))
