/* Общие настройки графиков Plotly и мелкие утилиты для страниц сайта. */
(function (global) {
  "use strict";

  var T = {};

  T.colors = {
    path: "rgba(70,110,180,0.30)",
    good: "#2d6cdf",
    good2: "#1f9e89",
    bad: "#c0392b",
    accent: "#e67e22",
    violet: "#7d3cbf",
    dark: "#22303f",
    grey: "#8a94a3",
    grid: "rgba(120,130,145,0.22)",
    band: "rgba(45,108,223,0.13)",
    zero: "rgba(60,70,85,0.55)"
  };
  T.seq = ["#2d6cdf", "#e67e22", "#1f9e89", "#c0392b", "#7d3cbf", "#b7950b", "#546e7a"];

  T.config = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"]
  };

  T.font = "DejaVu Sans, Segoe UI, Arial, sans-serif";

  /* Базовый layout: белый фон, сетка, легенда под графиком. */
  T.layout = function (extra) {
    var base = {
      font: { family: T.font, size: 13, color: T.colors.dark },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      margin: { l: 62, r: 18, t: 34, b: 96 },
      hovermode: "closest",
      legend: {
        orientation: "h", yanchor: "top", y: -0.16, x: 0,
        font: { size: 11.5 }, bgcolor: "rgba(0,0,0,0)"
      },
      colorway: T.seq
    };
    var out = Object.assign({}, base, extra || {});
    out.margin = Object.assign({}, base.margin, out.margin);
    out.legend = Object.assign({}, base.legend, out.legend);
    var ax = {
      showgrid: true, gridcolor: T.colors.grid, zeroline: false,
      linecolor: T.colors.grey, linewidth: 1, showline: true,
      tickfont: { size: 12 }, automargin: true
    };
    if (!out.xaxis) out.xaxis = Object.assign({}, ax);
    if (!out.yaxis) out.yaxis = Object.assign({}, ax);
    return out;
  };

  T.plot = function (id, traces, layout, config) {
    var el = document.getElementById(id);
    if (!el) return Promise.resolve();
    return Plotly.react(el, traces, T.layout(layout), Object.assign({}, T.config, config || {}));
  };

  /* Дебаунс: тяжёлые пересчёты запускаются после паузы в движении ползунка. */
  T.debounce = function (fn, ms) {
    var t = null;
    return function () {
      var args = arguments, self = this;
      clearTimeout(t);
      t = setTimeout(function () { fn.apply(self, args); }, ms || 80);
    };
  };

  /* Привязка всех input/select внутри контейнера к одному колбэку.
     Значения читаются из data-type: int | float | text. */
  T.bind = function (container, cb, ms) {
    var inputs = Array.prototype.slice.call(
      container.querySelectorAll("input, select"));
    function values() {
      var out = {};
      inputs.forEach(function (el) {
        var v;
        if (el.type === "range" || el.type === "number") {
          v = parseFloat(el.value);
          if ((el.dataset.type || "int") === "int") v = Math.round(v);
        } else if (el.type === "checkbox") {
          v = el.checked;
        } else {
          v = el.value;
        }
        out[el.id] = v;
        var o = document.getElementById(el.id + "-out");
        if (o) {
          var dec = el.dataset.fmt ? parseInt(el.dataset.fmt, 10) : 0;
          o.textContent = (typeof v === "number" && dec > 0)
            ? v.toFixed(dec).replace(".", ",") : String(v).replace(".", ",");
        }
      });
      return out;
    }
    var run = T.debounce(function () { cb(values()); }, ms || 90);
    inputs.forEach(function (el) {
      el.addEventListener("input", run);
      el.addEventListener("change", run);
    });
    return { values: values, run: function () { cb(values()); } };
  };

  /* Номер с запятой в качестве десятичного разделителя. */
  T.num = function (x, d) {
    if (!isFinite(x)) return "—";
    if (d === undefined) d = 3;
    return Number(x).toFixed(d).replace(".", ",");
  };

  /* Целиком записанное число для подписи оси: 100, 200, 500, 1000, 0.02. */
  T.tickLabel = function (v) {
    var a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1e-4 && a < 1e7) {
      return String(Number(v.toPrecision(12)));
    }
    var e = Math.floor(Math.log(a) / Math.LN10);
    var m = v / Math.pow(10, e);
    return String(Number(m.toPrecision(6))) + "e" + e;
  };

  /* Подписи для логарифмической оси без обозначения 10^k.
   * Широкий диапазон — числа 1, 2, 5 в каждой декаде (100, 200, 500, 1000);
   * узкий (меньше восьми раз) — обычный равномерный шаг, иначе на оси
   * остаётся одна-две подписи. */
  T.plainLog = function (lo, hi) {
    if (!(lo > 0) || !(hi > lo) || !isFinite(lo) || !isFinite(hi)) return {};
    var vals = [];
    if (hi / lo <= 5) {
      var span = hi - lo;
      var raw = span / 5;
      var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
      var norm = raw / mag, step;
      if (norm <= 1) step = 1 * mag;
      else if (norm <= 2) step = 2 * mag;
      else if (norm <= 5) step = 5 * mag;
      else step = 10 * mag;
      var start = Math.ceil(lo / step) * step;
      for (var v = start; v <= hi * 1.0000001; v += step) {
        vals.push(Number(v.toPrecision(12)));
      }
    } else {
      var e0 = Math.floor(Math.log(lo) / Math.LN10);
      var e1 = Math.ceil(Math.log(hi) / Math.LN10);
      var mult = [1, 2, 5];
      for (var e = e0; e <= e1; e++) {
        for (var q = 0; q < mult.length; q++) {
          var val = mult[q] * Math.pow(10, e);
          if (val >= lo * 0.9999999 && val <= hi * 1.0000001) vals.push(Number(val.toPrecision(12)));
        }
      }
    }
    if (!vals.length) return {};
    return { tickvals: vals, ticktext: vals.map(T.tickLabel) };
  };

  /* Формулы KaTeX; если CDN не загрузился, текст остаётся как есть. */
  T.math = function () {
    if (global.renderMathInElement) {
      global.renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false }
        ],
        throwOnError: false
      });
    }
  };

  global.T = T;
})(window);
