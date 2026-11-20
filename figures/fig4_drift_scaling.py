"""Рис. 4 (simpler_drift): масштабирование сублинейного сноса.

Слева -- профили среднего μ_{n,i} = c·i·n^{α−1}: после деления на n^α все
кривые ложатся на прямую i/n (условие \eqref{eq:mean} статьи).  Справа --
точная условная вероятность положительности моста при k ≈ μ_{n,n}; для шагов
±1 она равна безусловной вероятности и убывает как n^{α−1}, а не стремится
к единице.  Это ровно та величина, к которой сходится условная вероятность
в теореме (ср. beta_drift).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common.walks import (Bridge, parity_k, positivity_unconditioned,
                          sublinear_profile)
import common.style as st


def build() -> go.Figure:
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Профиль среднего после нормировки",
                                        "Положительность моста при k ≈ μₙ"))

    # ---------- панель A ----------
    for idx, (alpha, n) in enumerate([(0.55, 200), (0.55, 800),
                                      (0.75, 200), (0.75, 800),
                                      (0.90, 800)]):
        i = np.arange(1, n + 1)
        mu = np.cumsum(2 * sublinear_profile(n, 1.0, alpha) - 1)
        fig.add_trace(go.Scatter(x=i / n, y=mu / n ** alpha, mode="lines",
                                 name=f"α = {alpha}, n = {n}",
                                 line=dict(color=st.SEQ[idx % len(st.SEQ)],
                                           width=1.7,
                                           dash="solid" if n == 800 else "dot")),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="i / n",
                             line=dict(color=st.COLORS["dark"], width=1.2,
                                       dash="dash")), row=1, col=1)

    # ---------- панель B ----------
    ns = np.unique(np.round(np.logspace(np.log10(60), np.log10(4000), 11)
                            ).astype(int))
    for idx, alpha in enumerate([0.55, 0.70, 0.85]):
        pb, pu = [], []
        for n in ns:
            p = sublinear_profile(int(n), 1.0, alpha)
            mu = float(np.sum(2 * p - 1))
            k = parity_k(int(n), mu)
            pb.append(Bridge(p, k, 0).probability())
            pu.append(positivity_unconditioned(p, 0)[-1])
        fig.add_trace(go.Scatter(x=ns, y=pb, mode="lines+markers",
                                 name=f"мост, α = {alpha}",
                                 line=dict(color=st.SEQ[idx], width=2.0),
                                 marker=dict(size=4)), row=1, col=2)
        fig.add_trace(go.Scatter(x=ns, y=pu, mode="markers",
                                 name=f"безусловная, α = {alpha}",
                                 marker=dict(size=8, color=st.SEQ[idx],
                                             symbol="circle-open"),
                                 showlegend=False), row=1, col=2)
        # наклон n^{α−1}
        fig.add_trace(go.Scatter(x=ns, y=ns.astype(float) ** (alpha - 1.0),
                                 mode="lines", showlegend=False,
                                 line=dict(color=st.SEQ[idx], width=1.0,
                                           dash="dot")), row=1, col=2)
    fig.add_trace(go.Scatter(x=[ns[0], ns[-1]], y=[1, 1], mode="lines",
                             name="предел 1 (утверждение)",
                             line=dict(color=st.COLORS["bad"], width=1.4,
                                       dash="longdash")), row=1, col=2)

    fig.update_xaxes(title_text="i / n", row=1, col=1)
    fig.update_yaxes(title_text="μₙ,ᵢ / n^α", range=[0, 1.05], row=1, col=1)
    tv, tt = st.log_ticks(float(ns[0]), float(ns[-1]))
    yv, yt = st.log_ticks(0.02, 1.0)
    fig.update_xaxes(title_text="n", type="log", tickvals=tv, ticktext=tt,
                     row=1, col=2)
    fig.update_yaxes(title_text="P(Sᵢ > 0, i < n)", type="log",
                     tickvals=yv, ticktext=yt, row=1, col=2)
    return fig


if __name__ == "__main__":
    import os

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
    print("\n".join(st.save(build(), "fig4_drift_scaling", out,
                            presets=("paper_wide", "poster", "blog"))))
