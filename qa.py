"""Проверка собранных фигур без визуального просмотра.

Что проверяется:
  * для каждой фигуры есть pdf/png/svg во всех трёх пресетах;
  * в SVG нет NaN, None, Infinity, undefined (следствие ошибок в данных);
  * PNG непустые, размеры соответствуют пресету (ширина/высота в px × 3);
  * все текстовые подписи фигуры присутствуют в SVG (кириллица + юникод);
  * у фигуры нет пустых трасс (все x/y пусты).

Запуск: python qa.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import plotly.graph_objects as go
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
PRESETS = {"paper_wide": (7.1, 2.6), "poster": (9.6, 6.4), "blog": (11.5, 6.6)}
BAD = ("NaN", "None", "Infinity", "undefined")


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from figures import (fig1_typical_bridge, fig2_cramer_tilt,
                         fig3_excursion_factorization, fig4_drift_scaling,
                         fig5_sublinear_heatmap, fig6_graph_gallery,
                         fig7_exploration_front, fig8_poisson_bridge,
                         fig9_connectivity_gumbel, fig10_large_dev_exit)

    problems: list[str] = []
    built = []
    for mod in (fig1_typical_bridge, fig2_cramer_tilt,
                fig3_excursion_factorization, fig4_drift_scaling,
                fig5_sublinear_heatmap, fig6_graph_gallery,
                fig7_exploration_front, fig8_poisson_bridge,
                fig9_connectivity_gumbel, fig10_large_dev_exit):
        name = mod.__name__.split(".")[-1]
        fig = mod.build()
        built.append((name, fig))
        for preset, (win, hin) in PRESETS.items():
            base = OUT / f"{name}_{preset}"
            for ext in (".pdf", ".png", ".svg"):
                f = base.with_suffix(ext)
                if not f.exists() or f.stat().st_size < 2000:
                    problems.append(f"нет/пустой файл: {f.name}")
            png = base.with_suffix(".png")
            if png.exists():
                w, h = Image.open(png).size
                if abs(w - win * 96 * 3) > 3 or abs(h - hin * 96 * 3) > 3:
                    problems.append(f"{png.name}: размер {w}×{h}, "
                                    f"ожидался {int(win*96*3)}×{int(hin*96*3)}")
            svg = base.with_suffix(".svg")
            if svg.exists():
                text = html.unescape(svg.read_text(encoding="utf-8"))
                for tok in BAD:
                    if tok in text:
                        problems.append(f"{svg.name}: встречается {tok}")
                missing = [lab for lab in labels(fig) if lab not in text]
                if missing:
                    problems.append(f"{svg.name}: нет подписей {missing[:3]}")

    for name, fig in built:
        empty = [i for i, tr in enumerate(fig.data)
                 if hasattr(tr, "x") and tr.x is not None and len(tr.x) == 0]
        if empty:
            problems.append(f"{name}: пустые трассы {empty}")

    print(f"фигур: {len(built)}, файлов: "
          f"{len(list(OUT.glob('*')))}")
    if problems:
        print("\nПРОБЛЕМЫ:")
        for p in problems:
            print(" -", p)
        return 1
    print("QA пройден: файлы на месте, подписи и размеры в порядке")
    return 0


def labels(fig: go.Figure) -> list[str]:
    """Все подписи осей/заголовки/аннотации, которые обязаны попасть в SVG."""
    out: list[str] = []
    if fig.layout.title is not None and fig.layout.title.text:
        out.append(fig.layout.title.text)
    for key in fig.layout.to_plotly_json():
        if key.startswith(("xaxis", "yaxis")):
            ax = fig.layout[key]
            if ax.title is not None and ax.title.text:
                out.append(ax.title.text)
    for ann in fig.layout.annotations:
        if ann.text:
            out.append(ann.text)
    for tr in fig.data:
        if getattr(tr, "name", None) and tr.showlegend is not False:
            out.append(tr.name)
    return [s for s in out if s]


if __name__ == "__main__":
    raise SystemExit(main())
