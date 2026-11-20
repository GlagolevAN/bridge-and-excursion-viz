/* Точные вычисления для блуждания с шагами X_i = Y_i - 2, Y_i ~ Poisson(λ).
 * Прямой аналог viz/common/poisson_shift.py.
 *
 * При λ = 1 снос равен -1, дисперсия 1.  Конец k = θn в зоне больших
 * уклонений; преобразование Крамера снова даёт пуассоновский шаг,
 * λ' = θ + 2, поэтому условный закон моста совпадает с наклонённым.
 *
 * τ — последний момент i < n, когда S_i < 0.  Функция tauSurvival считает
 * P(τ >= t | S_n = k) точно, обратной динамикой по (i, h); хвост
 * пуассоновского шага обрезан там, где его масса < 1e-16.
 */
(function (global) {
  "use strict";

  var S = {};
  S.LAM = 1;
  S.SHIFT = 2;

  function lgamma(x) {
    var g = 7;
    var c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
      771.32342877765313, -176.61502916214059, 12.507343278686905,
      -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
    x -= 1;
    var a = c[0], t = x + g + 0.5, i;
    for (i = 1; i < g + 2; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }
  S.lgamma = lgamma;

  var cache = {};
  /* Распределение шага X = Y - 2: значения d и веса w (и их логарифмы). */
  S.kernel = function (lam) {
    var key = lam.toFixed(12);
    if (cache[key]) return cache[key];
    var dmax = Math.ceil(lam + 8 * Math.sqrt(lam) + 12);
    var d = new Int32Array(dmax + 1);
    var w = new Float64Array(dmax + 1);
    var lw = new Float64Array(dmax + 1);
    for (var j = 0; j <= dmax; j++) {
      d[j] = j - S.SHIFT;
      lw[j] = -lam + j * Math.log(lam) - lgamma(j + 1);
      w[j] = Math.exp(lw[j]);
    }
    cache[key] = { d: d, w: w, lw: lw };
    return cache[key];
  };

  S.poissonPmf = function (lam, m) {
    if (m < 0) return 0;
    if (lam <= 0) return m === 0 ? 1 : 0;
    return Math.exp(-lam + m * Math.log(lam) - lgamma(m + 1));
  };

  function logPois(lam, m) {
    if (m < 0) return -Infinity;
    if (lam <= 0) return m === 0 ? 0 : -Infinity;
    return -lam + m * Math.log(lam) - lgamma(m + 1);
  }

  /* dst[i] += w * src[i + s] (обратная свёртка: шаг из состояния x в x+s). */
  function addShifted(dst, src, s, w) {
    var H = dst.length, i;
    if (w === 0) return;
    if (s >= 0) {
      for (i = 0; i + s < H; i++) dst[i] += w * src[i + s];
    } else {
      var a = -s;
      for (i = a; i < H; i++) dst[i] += w * src[i - a];
    }
  }

  /* dst[i] += w * src[i - s] (прямая свёртка: шаг s из состояния x-s). */
  function addShiftedFwd(dst, src, s, w) {
    var H = dst.length, i;
    if (w === 0) return;
    if (s >= 0) {
      for (i = s; i < H; i++) dst[i] += w * src[i - s];
    } else {
      var a = -s;
      for (i = 0; i + a < H; i++) dst[i] += w * src[i + a];
    }
  }

  /* Точные выборки моста S_n = k, k = round(θn): m траекторий длины n+1. */
  S.sampleBridge = function (n, theta, m, rng) {
    var lam = theta + S.SHIFT;
    var K = S.kernel(lam);
    var k = Math.round(theta * n);
    var paths = [];
    var J = K.d.length;
    var lw = new Float64Array(J);
    var cw = new Float64Array(J);
    for (var q = 0; q < m; q++) {
      var path = new Int32Array(n + 1);
      var x = 0, t, j;
      for (t = 0; t < n; t++) {
        var rem = n - t - 1;
        var remLam = rem * lam;
        var base = S.SHIFT * rem;
        var mx = -Infinity, tot = 0;
        for (j = 0; j < J; j++) {
          lw[j] = K.lw[j] + logPois(remLam, k - x - K.d[j] + base);
          if (lw[j] > mx) mx = lw[j];
        }
        for (j = 0; j < J; j++) {
          tot += Math.exp(lw[j] - mx);
          cw[j] = tot;
        }
        var u = rng.next() * tot, idx = 0;
        while (idx < J - 1 && cw[idx] < u) idx++;
        x += K.d[idx];
        path[t + 1] = x;
      }
      paths.push(path);
    }
    return paths;
  };

  /* τ = последний i с S_i < 0 (0, если выхода не было). */
  S.tauOfPaths = function (paths) {
    var out = new Int32Array(paths.length);
    for (var q = 0; q < paths.length; q++) {
      var p = paths[q], last = 0;
      for (var i = 1; i < p.length; i++) if (p[i] < 0) last = i;
      out[q] = last;
    }
    return out;
  };

  /* Точная P(τ >= t | S_n = k) для t = 0..T. */
  S.tauSurvival = function (n, theta, T) {
    var lam = theta + S.SHIFT;
    var K = S.kernel(lam);
    var k = Math.round(theta * n);
    var H = k + S.SHIFT * n + 10;
    var keepFrom = Math.max(0, n - T);
    var rows = [];
    var i, j, t, m;
    for (i = 0; i <= Math.min(n, T); i++) rows.push(new Float64Array(H + 1));
    var prev = new Float64Array(H + 1);
    prev[k] = 1;
    if (keepFrom === 0) rows[0].set(prev);
    for (m = 1; m <= n; m++) {
      var cur = new Float64Array(H + 1);
      for (j = 0; j < K.d.length; j++) addShifted(cur, prev, K.d[j], K.w[j]);
      prev = cur;
      if (m >= keepFrom) rows[m - keepFrom].set(cur);
    }

    var xmin = -S.SHIFT * T - 1;
    var G = H - xmin + 1;
    var f = new Float64Array(G);
    f[0 - xmin] = 1;
    var den = S.poissonPmf(n * lam, k + S.SHIFT * n);
    var out = new Float64Array(T + 1);
    for (t = 0; t <= T; t++) {
      if (t > n) break;
      if (t > 0) {
        var nf = new Float64Array(G);
        for (j = 0; j < K.d.length; j++) addShiftedFwd(nf, f, K.d[j], K.w[j]);
        f = nf;
      }
      var g = rows[n - t - keepFrom];
      var joint = 0;
      for (i = Math.max(0, xmin); i <= H; i++) joint += f[i - xmin] * g[i];
      out[t] = Math.min(1, Math.max(0, 1 - joint / den));
    }
    out[0] = 1;
    return out;
  };

  global.PoissonShiftWalks = S;
  if (typeof module !== "undefined" && module.exports) module.exports = S;
})(typeof window !== "undefined" ? window : globalThis);
