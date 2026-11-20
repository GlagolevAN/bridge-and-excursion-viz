"""Рис. 9 (erdos_renyi): связность G(n, p) и предельный закон e^{−e^{−c}}.

Точная вероятность связности P_n(p) = (1 − (1−p)^{n−1})^{n−1} · P(экскурсия)
вычисляется ДП по пуассоновским весам.  При p = (log n + c)/n имеем
P_n → exp(−exp(−c)).  Слева -- коллапс по c, справа -- сдвиг порога по p.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common.erdos import connectivity
import common.style as st


def build() -> go.Figure:
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Коллапс: p = (log n + c) / n",
                                        "Порог по p"))

    # ---------- панель A ----------
    cs = np.linspace(-3.0, 3.0, 13)
    ns = [20, 50, 100, 200, 400]
    for idx, n in enumerate(ns):
        vals = [connectivity(n, (np.log(n) + c) / n) for c in cs]
        fig.add_trace(go.Scatter(x=cs, y=vals, mode="lines",
                                 name=f"n = {n}",
                                 line=dict(color=st.SEQ[idx % len(st.SEQ)],
                                           width=1.9)),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=cs, y=np.exp(-np.exp(-cs)), mode="lines",
                             name="e^{−e^{−c}}",
                             line=dict(color=st.COLORS["dark"], width=2.6,
                                       dash="dash")), row=1, col=1)

    # ---------- панель B ----------
    ps = np.linspace(0.005, 0.45, 60)
    for idx, n in enumerate([10, 20, 50, 100]):
        fig.add_trace(go.Scatter(x=ps, y=[connectivity(n, p) for p in ps],
                                 mode="lines", name=f"n = {n}",
                                 line=dict(color=st.SEQ[idx], width=1.9)),
                      row=1, col=2)
        pc = np.log(n) / n
        fig.add_trace(go.Scatter(x=[pc], y=[connectivity(n, pc)],
                                 mode="markers", showlegend=False,
                                 marker=dict(size=6, color=st.SEQ[idx],
                                             symbol="diamond")),
                      row=1, col=2)

    fig.update_xaxes(title_text="c", row=1, col=1)
    fig.update_yaxes(title_text="Pₙ((log n + c)/n)", range=[-0.02, 1.02],
                     row=1, col=1)
    fig.update_xaxes(title_text="p", row=1, col=2)
    fig.update_yaxes(title_text="Pₙ(p)", range=[-0.02, 1.02], row=1, col=2)
    return fig


if __name__ == "__main__":
    import os

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
    print("\n".join(st.save(build(), "fig9_connectivity_gumbel", out,
                            presets=("paper_wide", "poster", "blog"))))
