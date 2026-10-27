"""Shared styling and helpers for the Manim scenes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from manim import (
    VMobject, Text, ManimColor,
    interpolate_color,
)

DATA = Path(__file__).resolve().parent / "data"

BG = "#0e1116"
FG = "#e5e7eb"
MUTED = "#9ca3af"
GRID = "#374151"
STOCK = "#60a5fa"
BOND = "#34d399"
GOLD = "#fbbf24"
PURPLE = "#c084fc"
ORANGE = "#fb923c"
RED = "#f87171"
CYAN = "#22d3ee"
POS = "#f87171"   # positive correlation
NEG = "#60a5fa"   # negative correlation
MODE_COLORS = ["#60a5fa", "#f87171", "#34d399", "#fbbf24", "#c084fc",
               "#22d3ee", "#fb923c", "#f472b6", "#a3e635", "#e879f9",
               "#38bdf8", "#facc15"]
FONT = "Helvetica Neue"


def T(s: str, size: float = 28, color: str = FG, weight: str = "NORMAL") -> Text:
    return Text(s, font=FONT, font_size=size, color=color, weight=weight)


def load_history() -> pd.DataFrame:
    return pd.read_csv(DATA / "history.csv", index_col=0, parse_dates=True)


def load_json(name: str) -> dict:
    return json.loads((DATA / name).read_text())


def corr_color(rho: float) -> ManimColor:
    """Blue for negative, grey near zero, red for positive correlation."""
    rho = float(np.clip(rho, -1, 1))
    if rho >= 0:
        return interpolate_color(ManimColor(MUTED), ManimColor(POS), rho)
    return interpolate_color(ManimColor(MUTED), ManimColor(NEG), -rho)


def spring(p1, p2, coils: int = 7, width: float = 0.14, color: str = MUTED,
           stroke: float = 2.5) -> VMobject:
    """Zig-zag spring between two points (works for any direction)."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = p2 - p1
    L = np.linalg.norm(d)
    if L < 1e-6:
        return VMobject().set_points_as_corners([p1, p2])
    u = d / L
    n = np.array([-u[1], u[0], 0.0])
    lead = 0.12 * L
    pts = [p1, p1 + u * lead]
    m = 2 * coils
    for j in range(1, m):
        s = lead + (L - 2 * lead) * j / m
        pts.append(p1 + u * s + n * width * (1 if j % 2 else -1))
    pts += [p2 - u * lead, p2]
    v = VMobject(stroke_color=color, stroke_width=stroke)
    v.set_points_as_corners(pts)
    return v


def chain_modes(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Normal modes of N equal masses on a string fixed at both ends.

    K is the tridiagonal (2, -1) matrix. Returns (omega, V) with V's
    columns the orthonormal mode shapes sin(i k pi / (N+1)).
    """
    i = np.arange(1, N + 1)
    k = np.arange(1, N + 1)
    V = np.sin(np.outer(i, k) * np.pi / (N + 1))
    V /= np.linalg.norm(V, axis=0)
    omega = 2 * np.sin(k * np.pi / (2 * (N + 1)))
    return omega, V
