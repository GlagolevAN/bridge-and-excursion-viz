"""Рис. 5 (simpler_drift): карта точной положительности моста при k ≈ μₙ.

Снос сублинеен: μ_{n,i} = c·i·n^{α−1}, α ∈ (½, 1).  В координатах
(α, n) точная вероятность P(Sᵢ > 0, i < n | Sₙ = k) убывает по n при каждом
фиксированном α; линии уровня наклона n^{α−1} видны как параллельные
диагонали.  Панель справа показывает сечения и наклон n^{α−1}.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common.walks import Bridge, parity_k, sublinear_profile
import common.style as st

ALPHAS = np.round(np.linspace(0.55, 0.95, 9), 3)
NS = np.unique(np.round(np.logspace(np.log10(64), np.log10(2048), 9)).astype(int))


def build() -> go.Figure:
    grid = np.empty((len(ALPHAS), len(NS)))
    for ia, alpha in enumerate(ALPHAS):
        for in_, n in enumerate(NS):
            n = int(n)
            p = sublinear_profile(n, 1.0, float(alpha))
            k = parity_k(n, float(np.sum(2 * p - 1)))
            grid[ia, in_] = Bridge(p, k, 0).probability()

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.14,
        subplot_titles=("Точная вероятность P(мост положителен)",
                        "Сечения по n"),
        specs=[[{"type": "heatmap"}, {"type": "scatter"}]])

    fig.add_trace(go.Heatmap(
        z=grid, x=np.log10(NS), y=ALPHAS,
        colorscale=[[0.0, "#12355b"], [0.35, "#2d6cdf"],
                    [0.7, "#1f9e89"], [1.0, "#f4c95d"]],
        colorbar=dict(title=dict(text="P", font=dict(size=9)),
                      thickness=9, len=0.85, tickfont=dict(size=8)),
        hovertemplate="α = %{y:.2f}<br>n = 10^%{x:.2f}<br>P = %{z:.3f}<extra></extra>",
    ), row=1, col=1)

    # линия уровня P = 0.5
    cs = fig.data[0]
    zmin, zmax = float(grid.min()), float(grid.max())
    level = (0.5 - zmin) / (zmax - zmin)
    fig.add_trace(go.Contour(
        z=grid, x=np.log10(NS), y=ALPHAS, showscale=False,
        contours=dict(type="constraint", operation="=", value=0.5,
                      showlabels=False),
        line=dict(color="white", width=1.6),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    for idx, alpha in enumerate([0.6, 0.75, 0.9]):
        p_curve = []
        for n in NS:
            n = int(n)
            p = sublinear_profile(n, 1.0, alpha)
            k = parity_k(n, float(np.sum(2 * p - 1)))
            p_curve.append(Bridge(p, k, 0).probability())
        fig.add_trace(go.Scatter(x=NS, y=p_curve, mode="lines+markers",
                                 name=f"α = {alpha}",
                                 line=dict(color=st.SEQ[idx], width=2.0),
                                 marker=dict(size=5)), row=1, col=2)
        fig.add_trace(go.Scatter(x=NS, y=NS.astype(float) ** (alpha - 1.0),
                                 mode="lines", showlegend=False,
                                 line=dict(color=st.SEQ[idx], width=1.0,
                                           dash="dot")), row=1, col=2)

    fig.update_xaxes(title_text="lg n", row=1, col=1,
                     tickvals=np.log10(NS), ticktext=[str(int(n)) for n in NS])
    fig.update_yaxes(title_text="α", row=1, col=1)
    tv, tt = st.log_ticks(float(NS[0]), float(NS[-1]))
    yv, yt = st.log_ticks(0.04, 0.8)
    fig.update_xaxes(title_text="n", type="log", tickvals=tv, ticktext=tt,
                     row=1, col=2)
    fig.update_yaxes(title_text="P(Sᵢ > 0, i < n)", type="log",
                     tickvals=yv, ticktext=yt, row=1, col=2)
    return fig


if __name__ == "__main__":
    import os

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
    print("\n".join(st.save(build(), "fig5_sublinear_heatmap", out,
                            presets=("paper_wide", "poster", "blog"))))
