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

  /* P(S_i > r, 1 <= i <= n) для профиля lam. */
  P.positivity = function (lam, r, sigmas) {
    if (sigmas === undefined) sigmas = 8;
    var n = lam.length, i, j;
    var mu = 0, va = 0, hiMax = 8;
    for (i = 0; i < n; i++) {
      mu += lam[i] - 1;
      va += lam[i];
      var hh = Math.ceil(mu + sigmas * Math.sqrt(va) + 20);
      if (hh > hiMax) hiMax = hh;
    }
    var lo = Math.min(0, (r === null || r === undefined) ? 0 : r + 1) - 1;
    var off = -lo;
    var H = hiMax - lo + 1;
    var prev = new Float64Array(H), cur = new Float64Array(H);
    prev[off] = 1;
    var logscale = 0, prevLo = off, prevHi = off;
    var cmu = 0, cva = 0;
    for (i = 1; i <= n; i++) {
      cmu += lam[i - 1] - 1;
      cva += lam[i - 1];
      var w = sigmas * Math.sqrt(cva) + 20;
      var hLo = Math.max(lo, Math.floor(cmu - w));
      var hHi = Math.min(hiMax, Math.ceil(cmu + w));
      if (r !== null && r !== undefined) hLo = Math.max(hLo, r + 1);
      var aLo = hLo + off, aHi = hHi + off;
      if (aLo > aHi || aLo < 0 || aHi >= H) return 0;
      for (j = aLo; j <= aHi; j++) cur[j] = 0;
      var kern = P.kernel(lam[i - 1]);
      for (var z = 0; z < kern.d.length; z++) {
        var d = kern.d[z], wt = kern.w[z];
        if (!(wt > 0)) continue;
        var j0 = Math.max(prevLo, aLo - d);
        var j1 = Math.min(prevHi, aHi - d);
        for (j = j0; j <= j1; j++) cur[j + d] += wt * prev[j];
      }
      var mx = 0;
      for (j = aLo; j <= aHi; j++) if (cur[j] > mx) mx = cur[j];
      if (!(mx > 0)) return 0;
      for (j = aLo; j <= aHi; j++) cur[j] /= mx;
      logscale += Math.log(mx);
      var tmp = prev;
      prev = cur;
      cur = tmp;
      prevLo = aLo;
      prevHi = aHi;
    }
    var s = 0;
    for (j = prevLo; j <= prevHi; j++) s += prev[j];
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

  /* Базовый профиль как в ER: w_i ∝ (1-1/n)^{i-1}, Σw_i = n;
     λ_i = 1 + c w_i, средний снос c. */
  P.erdosProfile = function (n, c) {
    var p = 1 / n;
    var geom = (1 - Math.pow(1 - p, n)) / p;
    var lam = new Float64Array(n);
    var w = 1;
    for (var i = 0; i < n; i++) {
      lam[i] = 1 + c * n * w / geom;
      w *= (1 - p);
    }
    return lam;
  };

  /* ------------------------------------------------------------------
   * Мост для неоднородного шага X_i = Y_i - 1, Y_i ~ Poisson(lam[i]).
   * Барьер S_i > r стоит при 1 <= i < n; концы S_0 = 0 и S_n = k не
   * контролируются.  Строки нормируются на максимум, множители копятся
   * в логарифме.
   * ------------------------------------------------------------------ */
  function bounds(lam, k, r) {
    var n = lam.length, mu = 0, va = 0, i;
    for (i = 0; i < n; i++) { mu += lam[i] - 1; va += lam[i]; }
    var sd = Math.sqrt(va);
    var hasBarrier = (r !== null && r !== undefined);
    var rlo = hasBarrier ? r + 1 : 0;
    var rhi = hasBarrier ? r : 0;
    /* При барьере состояния h <= r при i >= 1 всё равно обнуляются,
       поэтому нижняя граница окна не должна уходить на 10σ вниз. */
    var lo = hasBarrier
      ? Math.floor(Math.min(0, k, rlo) - 2)
      : Math.floor(Math.min(0, k) - 10 * sd - 30);
    var hi = Math.ceil(Math.max(mu, k, rhi) + 10 * sd + 30);
    if (lo > k) lo = k - 1;
    if (hi < k) hi = k + 1;
    return { lo: lo, hi: hi };
  }

  P.bridgeDP = function (lam, k, r, keep) {
    var n = lam.length;
    if (k < -n) {
      return { rows: null, log: -Infinity, last: 0, n: n, off: 0,
               lo: 0, hi: 0, actLo: null, actHi: null };
    }
    var b = bounds(lam, k, r), lo = b.lo, hi = b.hi;
    var H = hi - lo + 1, off = -lo;
    var cumMu = new Float64Array(n + 1), cumVar = new Float64Array(n + 1);
    var i, j, q;
    for (i = 0; i < n; i++) {
      cumMu[i + 1] = cumMu[i] + lam[i] - 1;
      cumVar[i + 1] = cumVar[i] + lam[i];
    }
    var muN = cumMu[n], varN = cumVar[n];
    var actLo = new Int32Array(n + 1), actHi = new Int32Array(n + 1);
    actLo[0] = off;
    actHi[0] = off;

    var rows = null, flat = null, cur, prev;
    if (keep) {
      flat = new Float64Array((n + 1) * H);
      rows = new Array(n + 1);
      rows[0] = flat.subarray(0, H);
      rows[0][off] = 1.0;
      cur = rows[0];
    } else {
      cur = new Float64Array(H);
      prev = new Float64Array(H);
      cur[off] = 1.0;
    }
    var logscale = 0.0, prevLo = off, prevHi = off;
    for (i = 1; i <= n; i++) {
      /* Окно вокруг условного среднего моста: для независимых шагов
         Var(S_i | S_n=k) = v_i - v_i^2/v_n, E = μ_i + (v_i/v_n)(k-μ_n). */
      var cvar = cumVar[i] - cumVar[i] * cumVar[i] / varN;
      if (!(cvar > 0)) cvar = 0;
      var cmean = cumMu[i] + (varN > 0 ? (cumVar[i] / varN) * (k - muN) : 0);
      var w = 8 * Math.sqrt(cvar) + 20;
      var hLo = Math.max(lo, Math.floor(cmean - w));
      var hHi = Math.min(hi, Math.ceil(cmean + w));
      if (i < n) {
        if (r !== null && r !== undefined) hLo = Math.max(hLo, r + 1);
      } else {
        hLo = k;
        hHi = k;
      }
      var aLo = hLo + off, aHi = hHi + off;
      if (aLo > aHi || aLo < 0 || aHi >= H) {
        return { rows: rows, log: -Infinity, last: 0, n: n, off: off,
                 lo: lo, hi: hi, actLo: actLo, actHi: actHi };
      }
      actLo[i] = aLo;
      actHi[i] = aHi;
      if (keep) {
        prev = cur;
        cur = rows[i] = flat.subarray(i * H, (i + 1) * H);
      } else {
        var tmp = prev;
        prev = cur;
        cur = tmp;
        for (j = aLo; j <= aHi; j++) cur[j] = 0;
      }
      var kern = P.kernel(lam[i - 1]);
      var d = kern.d, ww = kern.w;
      for (q = 0; q < d.length; q++) {
        var s = d[q], wt = ww[q];
        if (!(wt > 0)) continue;
        var j0 = Math.max(prevLo, aLo - s);
        var j1 = Math.min(prevHi, aHi - s);
        for (j = j0; j <= j1; j++) cur[j + s] += wt * prev[j];
      }
      var mx = 0;
      for (j = aLo; j <= aHi; j++) if (cur[j] > mx) mx = cur[j];
      if (!(mx > 0)) {
        return { rows: rows, log: -Infinity, last: 0, n: n, off: off,
                 lo: lo, hi: hi, actLo: actLo, actHi: actHi };
      }
      for (j = aLo; j <= aHi; j++) cur[j] /= mx;
      logscale += Math.log(mx);
      prevLo = aLo;
      prevHi = aHi;
    }
    return { rows: rows, log: logscale, last: cur[off + k],
             n: n, off: off, lo: lo, hi: hi, actLo: actLo, actHi: actHi };
  };

  /* P(S_i > r, 1 <= i < n | S_n = k).  Знаменатель точный:
     S_n + n ~ Poisson(Σλ_i), поэтому P(S_n = k) = pmf(Σλ, n+k). */
  P.bridgeProb = function (lam, k, r) {
    var n = lam.length;
    if (k < -n) return 0;
    if (r === null || r === undefined) return 1;
    var dp = P.bridgeDP(lam, k, r, false);
    if (!(dp.last > 0) || !isFinite(dp.log)) return 0;
    var s = 0;
    for (var i = 0; i < n; i++) s += lam[i];
    var den = P.poissonPmf(s, n + k);
    if (!(den > 0)) return 0;
    var v = dp.last * Math.exp(dp.log) / den;
    if (!isFinite(v)) return 0;
    return Math.min(1, Math.max(0, v));
  };

  /* Точные выборки моста: обратный ход по сохранённым строкам ДП. */
  P.sampleBridge = function (lam, k, r, size, rng, dp) {
    var n = lam.length;
    if (!dp) dp = P.bridgeDP(lam, k, r, true);
    if (!dp.rows) return [];
    var out = [];
    for (var s = 0; s < size; s++) {
      var path = new Int32Array(n + 1);
      var h = k;
      path[n] = h;
      for (var i = n; i >= 1; i--) {
        var kern = P.kernel(lam[i - 1]);
        var prev = dp.rows[i - 1];
        var total = 0, chosen = [];
        for (var q = 0; q < kern.d.length; q++) {
          var hp = h - kern.d[q];
          var idx = hp + dp.off;
          if (idx < 0 || idx >= prev.length) continue;
          var ww = kern.w[q] * prev[idx];
          if (!(ww > 0)) continue;
          total += ww;
          chosen.push([hp, ww]);
        }
        if (!(total > 0)) {
          h = h + 1;
        } else {
          var u = rng.next() * total, acc = 0, pick = chosen[chosen.length - 1][0];
          for (var z = 0; z < chosen.length; z++) {
            acc += chosen[z][1];
            if (u <= acc) { pick = chosen[z][0]; break; }
          }
          h = pick;
        }
        path[i - 1] = h;
      }
      out.push(path);
    }
    return out;
  };

  /* Среднее и поточечные квантили моста по строкам ДП. */
  P.bridgeStats = function (dp, lam, k, r, qlo, qhi) {
    if (qlo === undefined) qlo = 0.05;
    if (qhi === undefined) qhi = 0.95;
    var n = dp.n, H = dp.hi - dp.lo + 1, off = dp.off;
    var aLo = dp.actLo, aHi = dp.actHi;
    var bwd = new Array(n + 1);
    for (var q = 0; q <= n; q++) bwd[q] = new Float64Array(H);
    bwd[n][off + k] = 1.0;
    for (var i = n - 1; i >= 0; i--) {
      var cur = bwd[i], nxt = bwd[i + 1];
      var kern = P.kernel(lam[i]);
      var l0 = aLo[i], h0 = aHi[i], l1 = aLo[i + 1], h1 = aHi[i + 1];
      for (var z = 0; z < kern.d.length; z++) {
        var d = kern.d[z], w = kern.w[z];
        if (!(w > 0)) continue;
        var j0 = Math.max(l0, l1 - d);
        var j1 = Math.min(h0, h1 - d);
        for (var j = j0; j <= j1; j++) cur[j] += w * nxt[j + d];
      }
      var mx = 0;
      for (var j4 = l0; j4 <= h0; j4++) if (cur[j4] > mx) mx = cur[j4];
      if (mx > 0) for (var j5 = l0; j5 <= h0; j5++) cur[j5] /= mx;
    }
    var mean = new Float64Array(n + 1);
    var loOut = new Float64Array(n + 1);
    var hiOut = new Float64Array(n + 1);
    for (var ii = 0; ii <= n; ii++) {
      var row = dp.rows[ii], bw = bwd[ii];
      var l = aLo[ii], hh = aHi[ii];
      var sum = 0, m = 0;
      for (var j6 = l; j6 <= hh; j6++) {
        var ww = row[j6] * bw[j6];
        if (ww > 0) { sum += ww; m += ww * (j6 - off); }
      }
      mean[ii] = sum > 0 ? m / sum : 0;
      var acc = 0, gotLo = false, gotHi = false;
      for (var j7 = l; j7 <= hh && sum > 0; j7++) {
        var w2 = row[j7] * bw[j7];
        if (!(w2 > 0)) continue;
        acc += w2 / sum;
        if (!gotLo && acc >= qlo) { loOut[ii] = j7 - off; gotLo = true; }
        if (!gotHi && acc >= qhi) { hiOut[ii] = j7 - off; gotHi = true; break; }
      }
      if (!gotHi) hiOut[ii] = 0;
    }
    return { mean: mean, lo: loOut, hi: hiOut };
  };

  global.PoissonWalks = P;
  if (typeof module !== "undefined" && module.exports) module.exports = P;
})(typeof window !== "undefined" ? window : globalThis);
