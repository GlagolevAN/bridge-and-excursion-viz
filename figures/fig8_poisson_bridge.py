"""Рис. 8 (erdos_renyi): пуассоновский мост и вероятность экскурсии.

При условии суммы независимые пуассоновские X_t с λ_t = q(1−p)^{t−1}
распределены полиномиально, поэтому мост S_k = Σ_{i≤k}(X_i − 1) выбирается
точно одним вызовом multinomial(n−1, λ/Σλ).  Справа -- точная вероятность
экскурсии (ДП по пуассоновским весам) против Монте-Карло.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common.erdos import excursion_probability, poisson_bridge
import common.style as st

N = 120
C = 1.0


def build() -> go.Figure:
    rng = np.random.default_rng(808)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Точные выборки моста, p = c/n",
                                        "P(экскурсия): ДП и Монте-Карло"))

    # ---------- панель A ----------
    p = C / N
    paths = poisson_bridge(N, p, 40, rng)
    k = np.arange(N + 1)
    for path in paths:
        fig.add_trace(go.Scatter(x=k, y=path, mode="lines",
                                 line=dict(color=st.COLORS["path"], width=1.0),
                                 showlegend=False, hoverinfo="skip"),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, N], y=[0, 0], mode="lines",
                             line=dict(color=st.COLORS["zero"], width=1.2,
                                       dash="dash"),
                             name="S = 0", showlegend=False), row=1, col=1)
    fig.add_annotation(x=0.04, y=0.93, xref="x domain", yref="y domain",
                       text=f"n = {N}, c = {C}", showarrow=False,
                       font=dict(size=9), row=1, col=1)

    # ---------- панель B ----------
    cs = np.linspace(0.15, 2.0, 13)
    for idx, n in enumerate([20, 50, 100, 200]):
        exact = [excursion_probability(n, c / n) for c in cs]
        fig.add_trace(go.Scatter(x=cs, y=exact, mode="lines",
                                 name=f"n = {n}",
                                 line=dict(color=st.SEQ[idx], width=2.0)),
                      row=1, col=2)
        mc = []
        for c in cs[::3]:
            smp = poisson_bridge(n, c / n, 3000, rng)
            mc.append(float(np.mean(smp[:, 1:n].min(axis=1) >= 0)))
        fig.add_trace(go.Scatter(x=cs[::3], y=mc, mode="markers",
                                 marker=dict(size=6, color=st.SEQ[idx],
                                             symbol="circle-open"),
                                 showlegend=False, hoverinfo="skip"),
                      row=1, col=2)

    fig.update_xaxes(title_text="k", row=1, col=1)
    fig.update_yaxes(title_text="Sₖ", row=1, col=1)
    fig.update_xaxes(title_text="c, где p = c / n", row=1, col=2)
    fig.update_yaxes(title_text="P(Sₖ ≥ 0, 1 ≤ k ≤ n−2)", range=[0, 1],
                     row=1, col=2)
    return fig


if __name__ == "__main__":
    import os

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
    print("\n".join(st.save(build(), "fig8_poisson_bridge", out,
                            presets=("paper_wide", "poster", "blog"))))
