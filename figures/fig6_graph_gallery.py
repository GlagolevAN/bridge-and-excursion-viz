"""Рис. 6 (erdos_renyi): галерея графов G(n, p) вокруг порога связности.

Порог: p_c = log n / n.  Слева p = 0.5·p_c (много малых компонент), в центре
p = p_c (появляется гигантская компонента, но связности ещё нет), справа
p = 1.6·p_c (граф почти наверняка связен).  Цвет вершины -- её компонента
связности, рёбра гигантской компоненты выделены.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import common.style as st

N = 24


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def graph_sample(n: int, p: float, rng: np.random.Generator):
    dsu = DSU(n)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
                dsu.union(u, v)
    comp = [dsu.find(i) for i in range(n)]
    sizes = {}
    for c in comp:
        sizes[c] = sizes.get(c, 0) + 1
    return edges, comp, sizes


def build() -> go.Figure:
    rng = np.random.default_rng(2026)
    pc = np.log(N) / N
    cases = [(0.5 * pc, "p = 0,5·p꜀"), (pc, "p = p꜀"), (1.6 * pc, "p = 1,6·p꜀")]

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.03,
                        subplot_titles=[c[1] for c in cases])
    theta = 2 * np.pi * np.arange(N) / N + np.pi / 2
    xs, ys = np.cos(theta), np.sin(theta)

    palette = ["#d5dbe3", "#9fb2c9", "#6f8fb5", "#4a6f9c", "#2d6cdf",
               "#e67e22", "#1f9e89", "#c0392b"]

    for col, (p, _) in enumerate(cases, start=1):
        edges, comp, sizes = graph_sample(N, p, rng)
        giant = max(sizes, key=sizes.get)
        # рёбра
        ex, ey = [], []
        for u, v in edges:
            ex += [xs[u], xs[v], None]
            ey += [ys[u], ys[v], None]
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                                 line=dict(color="rgba(90,105,125,0.45)",
                                           width=0.8),
                                 hoverinfo="skip", showlegend=False),
                      row=1, col=col)
        # вершины
        order = sorted(sizes, key=lambda c: -sizes[c])
        color_of = {c: palette[min(i, len(palette) - 1)]
                    for i, c in enumerate(order)}
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=9, color=[color_of[c] for c in comp],
                        line=dict(color="white", width=1.0)),
            text=[f"компонента {c}" for c in comp],
            hoverinfo="text", showlegend=False),
            row=1, col=col)
        fig.add_annotation(x=0.5, y=-0.10, xref="x domain", yref="y domain",
                           text=(f"компонент: {len(sizes)}, "
                                 f"гигант: {sizes[giant]}/{N}, "
                                 f"рёбер: {len(edges)}"),
                           showarrow=False, font=dict(size=8),
                           row=1, col=col)

    fig.update_xaxes(visible=False, range=[-1.2, 1.2])
    fig.update_yaxes(visible=False, range=[-1.35, 1.15])
    return fig


if __name__ == "__main__":
    import os

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
    print("\n".join(st.save(build(), "fig6_graph_gallery", out,
                            presets=("paper_wide", "poster", "blog"))))
