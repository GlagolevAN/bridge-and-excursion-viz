/* Точные вычисления для блуждания с шагами X_i = Y_i - 1,
 * Y_i ~ Poisson(λ_i).  Прямой аналог viz/common/poisson.py.
 *
 * Шаг принимает значения -1, 0, 1, 2, ...; пуассоновский хвост обрезан
 * там, где его масса < 1e-16, окно высот — с запасом 10σ.  Мост S_n = k
 * считается одним проходом ДП с поглощающим барьером h <= r; деление на
 * P(Poisson(nλ) = n+k) даёт условную вероятность.
 */
(function (global) {
  "use strict";

  var P = {};

  /* Пары (d, w): значения шага d = -1,0,1,... и их вероятности.
   * Ядро зависит только от λ, поэтому кэшируется. */
  var kernelCache = {};
  P.kernel = function (lam) {
    var key = lam.toFixed(12);
    if (kernelCache[key]) return kernelCache[key];
    var dmax = Math.ceil(lam + 8 * Math.sqrt(lam) + 12);
    var d = new Int32Array(dmax + 2);
    var w = new Float64Array(dmax + 2);
    var j;
    for (j = 0; j < d.length; j++) d[j] = j - 1;
    w[0] = Math.exp(-lam);
    for (j = 0; j <= dmax; j++) w[j + 1] = w[j] * lam / (j + 1);
    kernelCache[key] = { d: d, w: w };
    return kernelCache[key];
  };

  P.poissonPmf = function (lam, m) {
    if (m < 0) return 0;
    if (lam <= 0) return m === 0 ? 1 : 0;
    return Math.exp(-lam + m * Math.log(lam) - lgamma(m + 1));
  };

  /* log Γ(x) для целого x >= 0 (ланцзош, достаточно для pmf). */
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
  P.lgamma = lgamma;

  function convolve(prev, kern) {
    var H = prev.length, cur = new Float64Array(H);
    var d = kern.d, w = kern.w, i, s, j;
    for (i = 0; i < d.length; i++) {
      s = d[i];
      if (w[i] === 0) continue;
      if (s >= 0) {
        if (s < H) for (j = 0; j < H - s; j++) cur[j + s] += w[i] * prev[j];
      } else {
        s = -s;
        if (s < H) for (j = 0; j < H - s; j++) cur[j] += w[i] * prev[j + s];
      }
    }
    return cur;
  }

  /* P(S_i > r, 1 <= i < n | S_n = k) для k = 0..kmax. */
  P.bridgeCurve = function (n, lam, r, kmax) {
    if (kmax === undefined) kmax = Math.round(1.25 * n) + 4;
    var kern = P.kernel(lam);
    var hi = kmax + n;
    var lo = Math.min(0, r);
    var off = -lo;
    var H = hi - lo + 1;
    var prev = new Float64Array(H);
    prev[off] = 1;
    var logscale = 0, i, j, cur, mx;
    for (i = 1; i <= n; i++) {
      cur = convolve(prev, kern);
      if (i < n) {
        var cut = r + off + 1;
        for (j = 0; j < cut && j < H; j++) cur[j] = 0;
      }
      mx = 0;
      for (j = 0; j < H; j++) if (cur[j] > mx) mx = cur[j];
      if (!(mx > 0)) return new Float64Array(kmax + 1);
      for (j = 0; j < H; j++) cur[j] /= mx;
      logscale += Math.log(mx);
      prev = cur;
    }
    var scale = Math.exp(logscale);
    var out = new Float64Array(kmax + 1);
    for (var k = 0; k <= kmax; k++) {
      var den = P.poissonPmf(n * lam, n + k);
      if (den > 0) out[k] = Math.min(1, Math.max(0, prev[k + off] * scale / den));
    }
    return out;
  };

  /* P(S_i > r, 1 <= i < n) для профиля lam. */
  P.positivity = function (lam, r, sigmas) {
    if (sigmas === undefined) sigmas = 10;
    var n = lam.length, i, j;
    var mu = 0, va = 0, hiMax = 8;
    for (i = 0; i < n; i++) {
      mu += lam[i] - 1;
      va += lam[i];
      var hh = Math.ceil(mu + sigmas * Math.sqrt(va) + 20);
      if (hh > hiMax) hiMax = hh;
    }
    var lo = Math.min(0, r);
    var off = -lo;
    var H = hiMax - lo + 1;
    var prev = new Float64Array(H);
    prev[off] = 1;
    var logscale = 0;
    var cmu = 0, cva = 0;
    for (i = 1; i <= n; i++) {
      var kern = P.kernel(lam[i - 1]);
      var cur = convolve(prev, kern);
      if (i < n) {
        var cut = r + off + 1;
        for (j = 0; j < cut && j < H; j++) cur[j] = 0;
      }
      cmu += lam[i - 1] - 1;
      cva += lam[i - 1];
      var hi = Math.ceil(cmu + sigmas * Math.sqrt(cva) + 20);
      for (j = hi + off + 1; j < H; j++) cur[j] = 0;
      var mx = 0;
      for (j = 0; j < H; j++) if (cur[j] > mx) mx = cur[j];
      if (!(mx > 0)) return 0;
      for (j = 0; j < H; j++) cur[j] /= mx;
      logscale += Math.log(mx);
      prev = cur;
    }
    var s = 0;
    for (j = 0; j < H; j++) s += prev[j];
    return Math.min(1, Math.max(0, s * Math.exp(logscale)));
  };

  /* e^θ, при котором E_θ S_n = k:  (n+k)/Σλ_i. */
  P.tiltScale = function (lam, n, k) {
    var s = 0;
    for (var i = 0; i < lam.length; i++) s += lam[i];
    return (n + k) / s;
  };

  P.tiltedProfile = function (lam, n, k) {
    var sc = P.tiltScale(lam, n, k);
    var out = new Float64Array(lam.length);
    for (var i = 0; i < lam.length; i++) out[i] = lam[i] * sc;
    return out;
  };

  /* Распределение шага X = Y - 1, Y ~ Poisson(lam): P(X = j), j >= -1. */
  P.stepPmf = function (lam, jmin, jmax) {
    var kern = P.kernel(lam), out = new Float64Array(jmax - jmin + 1);
    for (var j = jmin; j <= jmax; j++) {
      var idx = j + 1;
      if (idx >= 0 && idx < kern.w.length) out[j - jmin] = kern.w[idx];
    }
    return out;
  };

  global.PoissonWalks = P;
  if (typeof module !== "undefined" && module.exports) module.exports = P;
})(typeof window !== "undefined" ? window : globalThis);
