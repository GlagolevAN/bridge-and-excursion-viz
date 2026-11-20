"""Рис. 10 (rec_struct, теорема о сопряжённой мере): где именно траектория
уходит ниже нуля, если конец лежит в зоне больших уклонений.

Шаги X_i = Y_i - 2, Y_i ~ Poisson(1) (снос -1, дисперсия 1).  Мост
S_n = k, k/n = θ > 0.  Слева — точные выборки моста при θ = 1: траектория
растёт со скоростью θ, потому что условный закон совпадает с наклонённым.
В середине — первые 30 шагов: красным выделен начальный участок до
последнего выхода ниже нуля.  Справа — точная P(τ >= t | S_n = k), где
τ = последний момент S_i < 0; хвост убывает экспоненциально, то есть
поздний выход — редкое большое уклонение.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import common.style as st
from common.poisson_shift import sample_bridge, tau_of_paths, tau_survival

N = 400
THETA = 1.0
NPATH = 24
SEED = 20261118
T = 40
THETAS = (0.5, 1.0, 2.0)
ZOOM = 30


def build() -> go.Figure:
    rng = np.random.default_rng(SEED)
    paths = sample_bridge(N, THETA, NPATH, rng)
    tau = tau_of_paths(paths)
    x = np.arange(N + 1)

    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.105,
        subplot_titles=("Мост при k = n", "Первые 30 шагов",
                        "Последний выход"))

    # ---------- панель A: весь мост ----------
    for j, path in enumerate(paths):
        fig.add_trace(go.Scatter(
            x=x, y=path, mode="lines",
            line=dict(color=st.COLORS["path"], width=1.0),
            name="траектории моста", showlegend=(j == 0), hoverinfo="skip"),
            row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0, N], y=[0, N], mode="lines",
        line=dict(color=st.COLORS["dark"], width=1.2, dash="dot"),
        name="наклон k/n = 1", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0, N], y=[0, 0], mode="lines",
        line=dict(color=st.COLORS["zero"], width=1.2, dash="dash"),
        showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_shape(type="rect", x0=0, x1=ZOOM, y0=-4, y1=46,
                  line=dict(color=st.COLORS["grey"], width=1, dash="dot"),
                  fillcolor="rgba(0,0,0,0)", row=1, col=1)

    # ---------- панель B: начало крупно ----------
    named = False
    for path, t in zip(paths, tau):
        fig.add_trace(go.Scatter(
            x=x[:ZOOM + 1], y=path[:ZOOM + 1], mode="lines",
            line=dict(color=st.COLORS["path"], width=1.2),
            showlegend=False, hoverinfo="skip"), row=1, col=2)
        if t == 0:
            continue
        sel = slice(0, t + 1)
        fig.add_trace(go.Scatter(
            x=x[sel], y=path[sel], mode="lines",
            line=dict(color=st.COLORS["bad"], width=2.6),
            name="выход ниже нуля", showlegend=not named,
            hoverinfo="skip"), row=1, col=2)
        named = True
        fig.add_trace(go.Scatter(
            x=[t], y=[path[t]], mode="markers",
            marker=dict(size=6, color=st.COLORS["bad"],
                        line=dict(color="white", width=1)),
            showlegend=False, hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[0, ZOOM], y=[0, 0], mode="lines",
        line=dict(color=st.COLORS["zero"], width=1.2, dash="dash"),
        showlegend=False, hoverinfo="skip"), row=1, col=2)

    # ---------- панель C: хвост последнего выхода ----------
    ts = np.arange(1, T + 1)
    for th, color in zip(THETAS, (st.SEQ[0], st.SEQ[1], st.SEQ[2])):
        sv = tau_survival(N, th, T)[1:]
        fig.add_trace(go.Scatter(
            x=ts, y=sv, mode="lines", name=f"k/n = {th:.1f}".replace(".", ","),
            line=dict(color=color, width=2.1)), row=1, col=3)

    fig.update_xaxes(title_text="i", row=1, col=1)
    fig.update_yaxes(title_text="Sᵢ", row=1, col=1, range=[-8, 412])
    fig.update_xaxes(title_text="i", row=1, col=2, range=[0, ZOOM])
    fig.update_yaxes(title_text="Sᵢ", row=1, col=2, range=[-4.5, 48])
    fig.update_xaxes(title_text="t", row=1, col=3, range=[1, T])
    fig.update_yaxes(title_text="P(τ ≥ t | Sₙ = k)", row=1, col=3,
                     type="log", range=[-14, 0.05],
                     tickvals=[1, 1e-3, 1e-6, 1e-9, 1e-12],
                     ticktext=["1", "10⁻³", "10⁻⁶", "10⁻⁹", "10⁻¹²"])
    return fig


if __name__ == "__main__":
    import os

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
    print("\n".join(st.save(build(), "fig10_large_dev_exit", out,
                            presets=("paper_wide", "poster", "blog"))))
