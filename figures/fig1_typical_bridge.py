"""Рис. 1 (rec_struct, теорема 1): типичный мост в зоне нормальных уклонений.

Слева -- ансамбль точных выборок моста при неоднородном сносе; справа --
сходимость условной вероятности положительности на растущем начальном
участке к безусловной.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common.walks import (Bridge, const_profile, parity_k,
                          positivity_unconditioned)
import common.style as st

N = 400
PHI = 0.0


def profile(n: int) -> np.ndarray:
    """Неоднородный снос: p_i = 0.65 + 0.2 sin(2 pi i / n)."""
    i = np.arange(1, n + 1)
    return 0.65 + 0.2 * np.sin(2 * np.pi * i / n + PHI)


def build() -> go.Figure:
    rng = np.random.default_rng(20261103)

    # ---------- левая панель: ансамбль мостов ----------
    p = profile(N)
    mu = np.cumsum(2 * p - 1)[-1]
    k = parity_k(N, mu + 0.9 * np.sqrt(N))
    bridge = Bridge(p, k, 0)
    paths = bridge.sample(14, rng)
    mean = bridge.mean_path()
    q09 = bridge.band(0.9)

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                        subplot_titles=("Типичный мост", "Положительность на [0, T]"))

    x = np.arange(N + 1)
    for j, path in enumerate(paths):
        fig.add_trace(go.Scatter(x=x, y=path, mode="lines",
                                 line=dict(color=st.COLORS["path"], width=1.0),
                                 showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=mean, mode="lines", name="среднее моста",
                             line=dict(color=st.COLORS["good"], width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=q09, mode="lines", name="90%-квантиль",
                             line=dict(color=st.COLORS["accent"], width=1.6, dash="dot")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, N], y=[0, 0], mode="lines",
                             line=dict(color=st.COLORS["zero"], width=1, dash="dash"),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)

    # ---------- правая панель: сходимость по n ----------
    sizes = [100, 200, 400, 800]
    n_mc = 3000
    for idx, n in enumerate(sizes):
        pn = profile(n)
        mun = np.cumsum(2 * pn - 1)[-1]
        kn = parity_k(n, mun + 0.9 * np.sqrt(n))
        br = Bridge(pn, kn, 0)
        smp = br.sample(n_mc, rng, restricted=False)
        # g(T) = P(S_i > 0, i <= T | S_n = k)
        cond = np.minimum.accumulate(smp[:, 1:], axis=1) > 0
        g = cond.mean(axis=0)
        T = np.arange(1, n + 1)
        fig.add_trace(go.Scatter(x=T, y=g, mode="lines", name=f"n = {n}",
                                 line=dict(color=st.SEQ[idx], width=1.8)),
                      row=1, col=2)

    rho = positivity_unconditioned(const_profile(4000, 0.3), 0)
    fig.add_trace(go.Scatter(x=[1, 800], y=[rho[4000], rho[4000]], mode="lines",
                             name="предел ρ ≈ 0.30",
                             line=dict(color=st.COLORS["dark"], width=1.4, dash="dash")),
                  row=1, col=2)

    fig.update_xaxes(title_text="i", row=1, col=1)
    fig.update_yaxes(title_text="Sᵢ", row=1, col=1)
    tv, tt = st.log_ticks(1, 800)
    fig.update_xaxes(title_text="T", type="log", tickvals=tv, ticktext=tt,
                     row=1, col=2, range=[0, np.log10(800) + 0.02])
    fig.update_yaxes(title_text="P(Sᵢ > 0, i ≤ T | Sₙ = k)", row=1, col=2,
                     range=[0.25, 0.78])
    return fig

if __name__ == "__main__":
    import os
    import common.style as st

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
    print("\n".join(st.save(build(), "fig1_typical_bridge", out,
                            presets=("paper_wide", "poster", "blog"))))
