"""Рис. 7 (erdos_renyi): фронт обхода графа.

Процесс активных вершин A_{t+1} = A_t − 1 + Bin(n − t − A_t, p) живёт в
коридоре 0 ≤ A_t ≤ n − t.  Линия нулевого сноса a = n − t − 1/p отделяет
режим вымирания (p = c/n, c < 1) от режима роста гигантской компоненты
(c > 1).  В критическом случае c = 1 фронт задерживается на масштабе n^{2/3}.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common.erdos import bfs_front
import common.style as st

N = 600
CASES = [(0.6, "p = 0,6 / n"), (1.0, "p = 1 / n"), (1.8, "p = 1,8 / n")]


def build() -> go.Figure:
    rng = np.random.default_rng(4242)
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.06,
                        subplot_titles=[c[1] for c in CASES],
                        shared_yaxes=False)

    for col, (c, _) in enumerate(CASES, start=1):
        p = c / N
        for _ in range(12):
            run = bfs_front(N, p, rng)
            front = run["front"]
            t = np.arange(len(front))
            fig.add_trace(go.Scatter(x=t, y=front, mode="lines",
                                     line=dict(color=st.COLORS["path"], width=1.0),
                                     showlegend=False, hoverinfo="skip"),
                          row=1, col=col)
        # коридор и линия нулевого сноса
        tt = np.arange(0, N + 1)
        fig.add_trace(go.Scatter(x=tt, y=N - tt, mode="lines",
                                 line=dict(color=st.COLORS["grey"], width=1.0,
                                           dash="dot"),
                                 name="A = n − t", showlegend=False,
                                 hoverinfo="skip"), row=1, col=col)
        a0 = N - 1.0 / p
        fig.add_trace(go.Scatter(x=tt, y=N - tt - 1.0 / p, mode="lines",
                                 line=dict(color=st.COLORS["bad"], width=1.4,
                                           dash="dash"),
                                 name="нулевой снос", showlegend=False,
                                 hoverinfo="skip"), row=1, col=col)
        fig.add_annotation(x=0.04, y=0.92, xref="x domain", yref="y domain",
                           text=f"A₀ = n − 1/p = {a0:.0f}",
                           showarrow=False, font=dict(size=8,
                                                      color=st.COLORS["bad"]),
                           row=1, col=col)

    fig.update_xaxes(title_text="t", row=1, col=1)
    fig.update_yaxes(title_text="Aₜ", row=1, col=1)
    for col in (2, 3):
        fig.update_xaxes(title_text="t", row=1, col=col)
    fig.update_yaxes(range=[0, N], row=1, col=1)
    fig.update_yaxes(range=[0, N], row=1, col=2)
    fig.update_yaxes(range=[0, N], row=1, col=3)
    return fig


if __name__ == "__main__":
    import os

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
    print("\n".join(st.save(build(), "fig7_exploration_front", out,
                            presets=("paper_wide", "poster", "blog"))))
