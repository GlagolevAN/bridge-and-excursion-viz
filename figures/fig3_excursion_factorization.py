"""Рис. 3 (rec_struct, теорема 3): факторизация экскурсии по среднему сечению.

Двухблочный снос (+c, затем -c) даёт типичную траекторию-«горб».  Теорема
утверждает, что при условии S_m = k две половины асимптотически независимы,
поэтому вероятность остаться положительной на всём [0, n] распадается в
произведение вероятностей для половин.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common.walks import (Bridge, parity_k, positivity_unconditioned,
                          two_block_profile)
import common.style as st

C = 0.30


def build() -> go.Figure:
    rng = np.random.default_rng(31)

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.11,
                        subplot_titles=("Экскурсия и сечение в m",
                                        "Факторизация P(экскурсия)"))

    # ---------- панель A: ансамбль путей ----------
    n = 240
    m = n // 2
    p = two_block_profile(n, C)
    ktyp = parity_k(m, float(np.sum(2 * p[:m] - 1)))
    bridge = Bridge(p, 0, 0)
    paths = bridge.sample(18, rng)
    x = np.arange(n + 1)

    for path in paths:
        fig.add_trace(go.Scatter(x=x, y=path, mode="lines",
                                 line=dict(color=st.COLORS["path"], width=1.0),
                                 showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=bridge.mean_path(), mode="lines",
                             name="среднее моста",
                             line=dict(color=st.COLORS["good"], width=2.3)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[m, m], y=[0, float(paths.max())],
                             mode="lines", name="сечение Sₘ = k",
                             line=dict(color=st.COLORS["bad"], width=1.4, dash="dash")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[m], y=[ktyp], mode="markers",
                             marker=dict(size=8, color=st.COLORS["bad"]),
                             name=f"типичное k = {ktyp}", showlegend=False),
                  row=1, col=1)
    fig.add_annotation(x=m / n, y=0.94, xref="x domain", yref="y domain",
                       text=f"Sₘ = {ktyp}", showarrow=False,
                       font=dict(size=9, color=st.COLORS["bad"]), row=1, col=1)

    # ---------- панель B: сходимость произведения ----------
    sizes = [40, 80, 160, 320, 640]
    full, prod, half = [], [], []
    for nn in sizes:
        mm = nn // 2
        pp = two_block_profile(nn, C)
        kk = parity_k(mm, float(np.sum(2 * pp[:mm] - 1)))
        q1 = Bridge(pp[:mm], kk, 0).probability()
        q2 = Bridge(pp[mm:][::-1], kk, 0).probability()
        full.append(Bridge(pp, 0, 0).probability())
        prod.append(q1 * q2)
        half.append(positivity_unconditioned(pp[:mm], 0)[-1])

    fig.add_trace(go.Scatter(x=sizes, y=full, mode="lines+markers",
                             name="весь мост: P(Sᵢ > 0, i < n | Sₙ = 0)",
                             line=dict(color=st.COLORS["good"], width=2.4),
                             marker=dict(size=6)), row=1, col=2)
    fig.add_trace(go.Scatter(x=sizes, y=prod, mode="lines+markers",
                             name="произведение половин: q₁·q₂",
                             line=dict(color=st.COLORS["accent"], width=1.8,
                                       dash="dash"),
                             marker=dict(size=6, symbol="square")), row=1, col=2)
    fig.add_trace(go.Scatter(x=sizes, y=half, mode="lines",
                             name="безусловная q = P(Sᵢ > 0, i ≤ m)",
                             line=dict(color=st.COLORS["grey"], width=1.5,
                                       dash="dot")), row=1, col=2)
    rho = half[-1]
    fig.add_trace(go.Scatter(x=[sizes[0], sizes[-1]], y=[rho * rho, rho * rho],
                             mode="lines", name="предел q²",
                             line=dict(color=st.COLORS["dark"], width=1.2,
                                       dash="longdash"), showlegend=False),
                  row=1, col=2)

    fig.update_xaxes(title_text="Номер шага i", row=1, col=1)
    fig.update_yaxes(title_text="Sᵢ", row=1, col=1)
    tv, tt = st.log_ticks(sizes[0], sizes[-1])
    fig.update_xaxes(title_text="n", type="log", tickvals=tv, ticktext=tt,
                     row=1, col=2)
    fig.update_yaxes(title_text="Вероятность", range=[0.0, 0.2], row=1, col=2)
    return fig


if __name__ == "__main__":
    import os

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
    print("\n".join(st.save(build(), "fig3_excursion_factorization", out,
                            presets=("paper_wide", "poster", "blog"))))
