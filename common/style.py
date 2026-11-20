"""Оформление статических фигур: пресеты размеров и экспорт в pdf/png/svg.

Пресеты подобраны так, чтобы фигуры вставлялись в статью (колонка ~3.5in),
на постер (10in) и в блог (12in) без перевёрстки.
"""

from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go

PRESETS = {
    "paper": dict(width=3.55, height=2.55, font=9.5, title=10.5, tick=8.5,
                  legend=8.0, margin=dict(l=46, r=8, t=34, b=36)),
    "paper_wide": dict(width=7.1, height=2.6, font=10.5, title=11.5, tick=9.5,
                       legend=9.0, margin=dict(l=52, r=10, t=38, b=40)),
    "poster": dict(width=9.6, height=6.4, font=21, title=23, tick=18,
                   legend=17, margin=dict(l=86, r=28, t=70, b=74)),
    "blog": dict(width=11.5, height=6.6, font=14.5, title=17, tick=12.5,
                 legend=12.5, margin=dict(l=68, r=24, t=54, b=56)),
}

FONT = "DejaVu Sans, Arial, sans-serif"

COLORS = dict(
    path="rgba(70,110,180,0.30)",
    good="#2d6cdf",
    good2="#1f9e89",
    bad="#c0392b",
    accent="#e67e22",
    violet="#7d3cbf",
    dark="#22303f",
    grid="rgba(120,130,145,0.20)",
    band="rgba(45,108,223,0.15)",
    grey="#8a94a3",
    zero="rgba(60,70,85,0.55)",
)

SEQ = ["#2d6cdf", "#e67e22", "#1f9e89", "#c0392b", "#7d3cbf", "#b7950b", "#546e7a"]


def make(preset: str = "paper", title: str | None = None, *,
         legend: bool = True, grid: bool = True, xaxis: str | None = None,
         yaxis: str | None = None) -> go.Figure:
    """Пустая фигура с оформлением под выбранный носитель."""
    pr = PRESETS[preset]
    fig = go.Figure()
    fig.update_layout(
        width=int(pr["width"] * 96),
        height=int(pr["height"] * 96),
        font=dict(family=FONT, size=pr["font"], color=COLORS["dark"]),
        margin=pr["margin"],
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0.0,
                    font=dict(size=pr["legend"]), bgcolor="rgba(0,0,0,0)"),
        hovermode="closest",
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.0, xanchor="left",
                                     font=dict(size=pr["title"])))
    for ax, name in (("x", xaxis), ("y", yaxis)):
        upd = dict(showgrid=grid, gridcolor=COLORS["grid"], zeroline=False,
                   tickfont=dict(size=pr["tick"]), linecolor=COLORS["grey"],
                   linewidth=1, showline=True, mirror=False)
        if name:
            upd["title"] = dict(text=name, font=dict(size=pr["font"]))
        fig.update_layout({f"{ax}axis": upd})
    return fig


def log_ticks(lo: float, hi: float):
    """Подписи для логарифмической оси в виде обычных чисел: 100, 200, 500."""
    vals = []
    e0 = int(np.floor(np.log10(lo)))
    e1 = int(np.ceil(np.log10(hi)))
    for e in range(e0, e1 + 1):
        for m in (1, 2, 5):
            v = m * 10.0 ** e
            if lo * 0.999999 <= v <= hi * 1.000001:
                vals.append(v)
    texts = [f"{v:g}" for v in vals]
    return vals, texts


def legend_height(fig: go.Figure, preset: str = "paper") -> int:
    """Оценка высоты горизонтальной легенды по числу строк (без рендера).

    Длина строки оценивается как 0.62·fontsize на символ (кириллица шире
    латиницы); строки укладываются жадно в доступную ширину.  Нужна, чтобы
    нижнее поле не «съедало» подписи осей.
    """
    pr = PRESETS[preset]
    items = [str(t.name) for t in fig.data
             if t.showlegend is not False and getattr(t, "name", None)]
    if not items or not fig.layout.showlegend:
        return 0
    size = pr["legend"]
    avail = pr["width"] * 96 - pr["margin"]["l"] - pr["margin"]["r"]
    rows, used = 1, 0.0
    gap = 2.2 * size
    for text in items:
        w = 0.62 * size * len(text) + gap
        if used + w > avail and used > 0:
            rows += 1
            used = w
        else:
            used += w
    return int(rows * size * 1.55 + 6)


def style_figure(fig: go.Figure, preset: str = "paper") -> go.Figure:
    """Накладывает пресет на уже собранную фигуру (оси, шрифты, поля)."""
    pr = PRESETS[preset]
    margin = dict(pr["margin"])
    margin["b"] += legend_height(fig, preset)
    fig.update_layout(
        width=int(pr["width"] * 96),
        height=int(pr["height"] * 96),
        font=dict(family=FONT, size=pr["font"], color=COLORS["dark"]),
        margin=margin,
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=SEQ,
        legend=dict(orientation="h", yanchor="top", y=-0.155, x=0.0,
                    font=dict(size=pr["legend"]), bgcolor="rgba(0,0,0,0)"),
    )
    if fig.layout.title is not None and fig.layout.title.text:
        fig.layout.title.font.size = pr["title"]
    for key in list(fig.layout.to_plotly_json().keys()):
        if key.startswith(("xaxis", "yaxis")):
            ax = fig.layout[key]
            ax.showgrid = True
            ax.gridcolor = COLORS["grid"]
            ax.zeroline = False
            ax.linecolor = COLORS["grey"]
            ax.showline = True
            ax.tickfont.size = pr["tick"]
            if ax.title is not None and ax.title.text:
                ax.title.font.size = pr["font"]
    for ann in fig.layout.annotations:
        ann.font.size = pr["title"] - 0.5
    return fig


def save(fig: go.Figure, name: str, outdir: str,
         presets=("paper", "poster", "blog")) -> list[str]:
    """Сохраняет фигуру в pdf, png (300 dpi) и svg для каждого пресета."""
    os.makedirs(outdir, exist_ok=True)
    written = []
    for preset in presets:
        pr = PRESETS[preset]
        style_figure(fig, preset)
        w, h = int(pr["width"] * 96), int(pr["height"] * 96)
        suffix = "" if preset == "paper" else f"_{preset}"
        base = os.path.join(outdir, f"{name}{suffix}")
        fig.write_image(base + ".pdf", width=w, height=h)
        fig.write_image(base + ".png", width=w, height=h, scale=3)
        fig.write_image(base + ".svg", width=w, height=h)
        written += [base + ".pdf", base + ".png", base + ".svg"]
    return written
