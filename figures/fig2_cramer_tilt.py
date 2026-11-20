"""Рис. 2 (rec_struct, теорема о сопряжённой мере): преобразование Крамера
для блуждания с шагами X_i = Poisson(1) - 1.

Слева — как наклон меняет распределение шага: λ -> λ' = λe^θ.  Справа —
условная положительность моста S_n = k и безусловная положительность
наклонённого блуждания; равенство предельное, зазор O(n^{-1/2}) виден
как небольшое расхождение кривых.  Третья, серая кривая — исходное
блуждание без наклона (одно число, не зависящее от k).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common.poisson import (bridge_curve, kernel, positivity, tilted_profile)
import common.style as st

N = 400
LAM = 1.0
X_TARGET = 0.56
JMAX = 9


def build() -> go.Figure:
    lam = np.full(N, LAM)
    k_target = int(round(X_TARGET * N))
    lam_tilt = float(tilted_profile(lam, N, k_target)[0])
    theta = float(np.log(lam_tilt / LAM))

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Сопряжение шага",
                                        "Положительность моста"))

    # ---------- панель A: распределение шага ----------
    j = np.arange(-1, JMAX + 1)
    w0 = kernel(LAM)[1][:j.size]
    w1 = kernel(lam_tilt)[1][:j.size]
    fig.add_trace(go.Bar(x=j, y=w0, showlegend=False,
                         marker=dict(color=st.COLORS["grey"], opacity=0.55)),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=j, y=w1, showlegend=False,
                         marker=dict(color=st.COLORS["accent"], opacity=0.85)),
                  row=1, col=1)
    # Бары не попадают в легенду (plotly 5.24 + kaleido даёт class
    # "legendundefined"), поэтому подписи серий — невидимые маркеры.
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(symbol="square", size=10,
                                         color=st.COLORS["grey"]),
                             name="до наклона: λ = 1"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(symbol="square", size=10,
                                         color=st.COLORS["accent"]),
                             name=f"после: λ′ = {lam_tilt:.2f}"), row=1, col=1)
    fig.add_annotation(x=0.03, y=0.93, xref="x domain", yref="y domain",
                       text=f"k/n = {X_TARGET}, θ = {theta:.2f}",
                       showarrow=False, font=dict(size=10,
                                                  color=st.COLORS["dark"]),
                       row=1, col=1)

    # ---------- панель B: положительность ----------
    kmax = int(round(1.25 * N)) + 4
    bridge = bridge_curve(N, LAM, 0, kmax)
    xs, pb, pt = [], [], []
    for x in np.arange(0.1, 1.1401, 0.08):
        k = int(round(x * N))
        xs.append(k / N)
        pb.append(bridge[k])
        pt.append(positivity(tilted_profile(lam, N, k), 0))
    orig = positivity(lam, 0)

    fig.add_trace(go.Scatter(x=xs, y=pb, mode="lines+markers",
                             name="мост: P(Sᵢ > 0, i < n | Sₙ = k)",
                             line=dict(color=st.COLORS["good"], width=2.4),
                             marker=dict(size=4)), row=1, col=2)
    fig.add_trace(go.Scatter(x=xs, y=pt, mode="lines",
                             name="наклонённое: P_θ(Sᵢ > 0, i < n)",
                             line=dict(color=st.COLORS["accent"], width=1.8,
                                       dash="dash")), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0.0, 1.2], y=[orig, orig], mode="lines",
                             name="исходное: P(Sᵢ > 0, i < n)",
                             line=dict(color=st.COLORS["grey"], width=1.4,
                                       dash="dot")), row=1, col=2)
    fig.add_trace(go.Scatter(x=[X_TARGET], y=[bridge[k_target]], mode="markers",
                             marker=dict(size=9, color=st.COLORS["dark"],
                                         symbol="circle-open", line=dict(width=2)),
                             name="выделенный k/n", showlegend=False),
                  row=1, col=2)

    fig.update_xaxes(title_text="Значение шага j", dtick=1, row=1, col=1)
    fig.update_yaxes(title_text="P(Xᵢ = j)", row=1, col=1)
    fig.update_xaxes(title_text="x = k / n", row=1, col=2)
    fig.update_yaxes(title_text="Вероятность положительности",
                     range=[0, 1], row=1, col=2)
    fig.update_layout(barmode="overlay")
    return fig


if __name__ == "__main__":
    import os

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
    print("\n".join(st.save(build(), "fig2_cramer_tilt", out,
                            presets=("paper_wide", "poster", "blog"))))
