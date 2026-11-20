# bridge-and-excursion-viz

Иллюстрации к статьям о мостах и экскурсиях: десять статических фигур
(PDF + PNG 300 dpi + SVG) и интерактивный сайт с тремя страницами.
Все подписи и тексты — по-русски.

Сайт: **https://glagolevan.github.io/bridge-and-excursion-viz/**

## Что лежит в репозитории

| каталог | содержимое |
|---------|------------|
| `out/` | готовые файлы: 90 = 10 фигур × 3 пресета (`paper`, `poster`, `blog`) × 3 формата (pdf/png/svg) |
| `docs/` | статический сайт, публикуется GitHub Pages (корень сайта) |
| `figures/` | скрипты сборки фигур, у каждого `build()` и `__main__` |
| `common/` | общие ядра: пресеты и сохранение, точный мост, наклон, хвосты |

Сайт не требует сборки: `docs/index.html` открывается как есть. Plotly.js
и KaTeX подключены с CDN; без сети графики не построятся, но текст и
формулы останутся. Интерактивные расчёты выполняются в браузере тем же
точным ДП, ГСЧ — детерминированный mulberry32 (`docs/assets/rng.js`),
поэтому результат воспроизводится.

## Сборка фигур

```bash
pip install -r requirements.txt
python make_all.py              # все фигуры + проверки
python make_all.py fig3 fig9    # только выбранные
```

Зависимости зафиксированы: `plotly==5.24.1`, `kaleido==0.2.1` (несёт
собственный headless-браузер, системный Chrome не нужен), `numpy`,
`scipy`, `pillow`. Python в CI не нужен: готовые PNG уже лежат в
репозитории. Фигуры пишутся в `out/`; запускать модулем из корня
репозитория (`python -m figures.fig3_excursion_factorization`) — иначе
не найдётся пакет `common`.

После пересборки картинки каталога нужно обновить:

```bash
cp out/*_blog.png docs/figures/
```

## Проверки без визуального просмотра

```bash
python qa.py
```

`qa.py` для каждой фигуры и пресета проверяет: наличие и размер
`pdf/png/svg`; размеры PNG (пресет × 96 × 3); отсутствие `NaN`, `None`,
`Infinity`, `undefined` в SVG; наличие ключевых подписей; непустые трассы.
Текущий результат — «фигур: 10, файлов: 90 / QA пройден».

Ограничение: QA не умеет оценивать композицию (перекрытия подписей, поля,
читаемость). Перед вставкой в статью фигуры нужно посмотреть глазами.

## Локальный просмотр сайта

```bash
cd docs
python3 -m http.server 8000
# http://localhost:8000/
```

## Публикация

GitHub Pages настроен на ветку `main`, каталог `/docs`
(**Settings → Pages → Deploy from a branch**). Отдельный workflow не
нужен: сайт статический, сборка при публикации отсутствует. Достаточно
запушить изменения в `main`.

## Структура

```
.
├── common/
│   ├── style.py            пресеты, цвета, сохранение pdf/png/svg
│   ├── walks.py            точный мост, профили сноса, положительность (±1)
│   ├── poisson.py          точный мост и наклон для шагов Poisson(1)−1
│   ├── poisson_shift.py    мост, выборка и хвост τ для шагов Poisson(1)−2
│   └── erdos.py            обход, пуассоновский мост, связность
├── figures/                fig1…fig10
├── out/                    готовые pdf/png/svg
├── make_all.py             сборка всех фигур + qa.py
├── qa.py                   проверки без визуального просмотра
├── requirements.txt
└── docs/
    ├── index.html, rec_struct.html, simpler_drift.html, erdos_renyi.html
    ├── assets/{style.css,theme.js,rng.js,walks.js,poisson_walks.js,
    │           poisson_shift_walks.js,erdos.js}
    └── figures/*.png
```
