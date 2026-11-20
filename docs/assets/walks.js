/* Точные вычисления для неоднородного блуждания с шагами ±1.
 *
 * Прямой аналог viz/common/walks.py, но на JavaScript и без хранения
 * всего массива строк, когда нужна только вероятность.  Условие S_n = k
 * и запрет S_i <= r обрабатываются динамическим программированием по
 * (шаг, высота); выборки из моста точные, без rejection sampling.
 */
(function (global) {
  "use strict";

  var W = {};

  /* Ближайшее к target целое той же чётности, что и n. */
  W.parityK = function (n, target) {
    var k = Math.round(target);
    if ((((n - k) % 2) + 2) % 2 !== 0) k -= 1;
    return k;
  };

  W.constProfile = function (n, drift) {
    var d = Math.max(-0.98, Math.min(0.98, drift));
    var p = new Float64Array(n);
    for (var i = 0; i < n; i++) p[i] = 0.5 * (1 + d);
    return p;
  };

  /* μ_{n,i} = c·i·n^{α−1}: пошаговый снос c·n^{α−1}. */
  W.sublinearProfile = function (n, c, alpha) {
    return W.constProfile(n, c * Math.pow(n, alpha - 1));
  };

  /* Снос +c до m и −c после m. */
  W.twoBlockProfile = function (n, c, m) {
    if (m === undefined) m = Math.floor(n / 2);
    var p = new Float64Array(n);
    for (var i = 0; i < n; i++) p[i] = 0.5 * (1 + (i < m ? c : -c));
    return p;
  };

  /* ------------------------------------------------------------------
   * Прямой проход ДП.  Строки нормируются на максимум (иначе 2^{-n}
   * уходит в ноль), множители копятся в logscale.
   *
   * keep = true  -> храним все строки (нужно для выборки и среднего);
   * keep = false -> только последняя строка, память O(n).
   * ------------------------------------------------------------------ */
  /* Одна строка ДП: считается только по достижимому окну высот |h| <= i,
   * иначе половина работы уходит на нули.  Вне окна cur не трогаем: эти
   * ячейки больше никогда не читаются (следующая строка читает prev только
   * внутри окна, границы считаются отдельно). */
  function stepRow(prev, cur, i, pi, r, k, n, off) {
    var qi = 1 - pi;
    var lo = off - i, hi = off + i;
    cur[lo] = qi * prev[lo + 1];
    cur[hi] = pi * prev[hi - 1];
    for (var j = lo + 1; j < hi; j++) {
      cur[j] = pi * prev[j - 1] + qi * prev[j + 1];
    }

    var j2;
    if (i < n) {
      /* барьер: при 1 <= i < n разрешены только h > r */
      if (r !== null && r !== undefined) {
        var cut = off + r;
        for (j2 = lo; j2 <= cut && j2 <= hi; j2++) cur[j2] = 0;
      }
    } else {
      /* конечное условие S_n = k: остаётся одна высота */
      var keepAt = off + k;
      for (j2 = lo; j2 < hi + 1; j2++) if (j2 !== keepAt) cur[j2] = 0;
    }

    var mx = 0;
    for (j2 = lo; j2 <= hi; j2++) if (cur[j2] > mx) mx = cur[j2];
    if (mx <= 0) return -Infinity;
    for (j2 = lo; j2 <= hi; j2++) cur[j2] /= mx;
    return Math.log(mx);
  }

  function runDP(p, k, r, keep) {
    var n = p.length, off = n + 1, width = 2 * n + 3;
    var rows = null, flat = null, cur, prev;
    if (keep) {
      /* одна непрерывная матрица: строки — подмассивы, без n аллокаций */
      flat = new Float64Array((n + 1) * width);
      rows = new Array(n + 1);
      rows[0] = flat.subarray(0, width);
      rows[0][off] = 1.0;
      cur = rows[0];
    } else {
      cur = new Float64Array(width);
      prev = new Float64Array(width);
      cur[off] = 1.0;
    }
    var logscale = 0.0;

    for (var i = 1; i <= n; i++) {
      var pi = p[i - 1];
      if (keep) {
        prev = cur;
        cur = rows[i] = flat.subarray(i * width, (i + 1) * width);
      } else {
        var tmp = prev;
        prev = cur;
        cur = tmp;
      }
      var lg = stepRow(prev, cur, i, pi, r, k, n, off);
      if (lg === -Infinity) {
        return { rows: rows, log: -Infinity, last: 0, n: n, off: off };
      }
      logscale += lg;
    }
    return { rows: rows, log: logscale, last: cur[off + k], n: n, off: off };
  }

  /* P(S_i > r, 1 <= i < n | S_n = k) — точно. */
  W.bridgeProb = function (p, k, r) {
    var n = p.length;
    if ((((n - k) % 2) + 2) % 2 !== 0) return 0;
    if (r === null || r === undefined) {
      if (Math.abs(k) > n) return 0;
      return 1;
    }
    /* Концы не контролируются: S_0 = 0 и S_n = k могут быть <= r;
       барьер стоит только при 1 <= i < n. */
    var pos = runDP(p, k, r, false);
    var all = runDP(p, k, null, false);
    if (!(all.last > 0) || !(pos.last > 0)) return 0;
    var v = pos.last / all.last;
    if (!isFinite(v)) return 0;
    return Math.exp(Math.log(v) + pos.log - all.log);
  };

  /* Строки ДП для условного моста (для выборки и среднего). */
  W.bridgeRows = function (p, k, r) {
    return runDP(p, k, r, true);
  };

  /* size независимых точных выборок моста; результат — массив путей. */
  W.sampleBridge = function (p, k, r, size, rng, rows) {
    var n = p.length, off = n + 1;
    var dp = rows || runDP(p, k, r, true);
    if (!dp.rows) dp = runDP(p, k, r, true);
    var out = [];
    for (var s = 0; s < size; s++) {
      var path = new Int32Array(n + 1);
      var h = k;
      path[n] = h;
      for (var i = n; i >= 1; i--) {
        var prev = dp.rows[i - 1];
        var pi = p[i - 1];
        var up = pi * prev[h - 1 + off];
        var dn = (1 - pi) * prev[h + 1 + off];
        var tot = up + dn;
        if (tot <= 0) { h = h + 1; }
        else if (rng.next() * tot < up) { h = h - 1; }
        else { h = h + 1; }
        path[i - 1] = h;
      }
      out.push(path);
    }
    return out;
  };

  /* Среднее и поточечные квантили высоты моста по строкам ДП.
   *
   * Прямые строки ДП не conditioned на S_n = k: условие наложено только в
   * последней строке, поэтому их среднее — среднее безусловного блуждания,
   * которое на последнем шаге скачком попадает в k.  Домножаем прямые строки
   * на обратные веса B_i(h) = P(дойти из (i, h) в (n, k) с учётом запрета) и
   * нормируем произведение по h — это и есть маргинал моста. */
  W.bridgeStats = function (dp, p, k, r, qlo, qhi) {
    if (qlo === undefined) qlo = 0.05;
    if (qhi === undefined) qhi = 0.95;
    var n = dp.n, off = dp.off, width = 2 * n + 3;

    var bwd = new Array(n + 1);
    for (var q = 0; q <= n; q++) bwd[q] = new Float64Array(width);
    bwd[n][off + k] = 1.0;
    for (var i = n - 1; i >= 0; i--) {
      var pi = p[i], qi = 1 - pi, nxt = bwd[i + 1], cur = bwd[i];
      var lo = off - i, hi = off + i;
      var cut = (i >= 1 && r !== null && r !== undefined) ? off + r : lo - 1;
      for (var j = lo; j <= hi; j++) {
        if (j <= cut) { cur[j] = 0; continue; }
        cur[j] = pi * nxt[j + 1] + qi * nxt[j - 1];
      }
      var mx = 0;
      for (var j2 = lo; j2 <= hi; j2++) if (cur[j2] > mx) mx = cur[j2];
      if (mx > 0) for (var j3 = lo; j3 <= hi; j3++) cur[j3] /= mx;
    }

    var mean = new Float64Array(n + 1);
    var loOut = new Float64Array(n + 1);
    var hiOut = new Float64Array(n + 1);
    for (var ii = 0; ii <= n; ii++) {
      var row = dp.rows[ii], bw = bwd[ii];
      var s = 0, m = 0;
      for (var j4 = 0; j4 < width; j4++) {
        var w = row[j4] * bw[j4];
        if (w > 0) { s += w; m += w * (j4 - off); }
      }
      mean[ii] = s > 0 ? m / s : 0;
      var acc = 0, gotLo = false, gotHi = false;
      for (var j5 = 0; j5 < width && s > 0; j5++) {
        var ww = row[j5] * bw[j5];
        if (!(ww > 0)) continue;
        acc += ww / s;
        if (!gotLo && acc >= qlo) { loOut[ii] = j5 - off; gotLo = true; }
        if (!gotHi && acc >= qhi) { hiOut[ii] = j5 - off; gotHi = true; break; }
      }
      if (!gotHi) hiOut[ii] = 0;
    }
    return { mean: mean, lo: loOut, hi: hiOut };
  };

  /* P(S_i > r, 1 <= i <= n) без условия на S_n — точно. */
  W.positivity = function (p, r) {
    var n = p.length, off = n + 1, width = 2 * n + 3;
    var cur = new Float64Array(width), prev = new Float64Array(width);
    cur[off] = 1.0;
    var logscale = 0.0;
    var j;
    for (var i = 1; i <= n; i++) {
      var tmp = prev;
      prev = cur;
      cur = tmp;
      var pi = p[i - 1], qi = 1 - pi;
      var lo = off - i, hi = off + i;
      cur[lo] = qi * prev[lo + 1];
      cur[hi] = pi * prev[hi - 1];
      for (j = lo + 1; j < hi; j++) {
        cur[j] = pi * prev[j - 1] + qi * prev[j + 1];
      }
      /* поглощающий барьер h <= r */
      var cut = off + r;
      for (j = lo; j <= cut && j <= hi; j++) cur[j] = 0;
      var mx = 0;
      for (j = lo; j <= hi; j++) if (cur[j] > mx) mx = cur[j];
      if (!(mx > 0)) return 0;
      for (j = lo; j <= hi; j++) cur[j] /= mx;
      logscale += Math.log(mx);
    }
    var sum = 0;
    for (j = off - n; j <= off + n; j++) sum += cur[j];
    if (!(sum > 0)) return 0;
    return Math.exp(Math.log(sum) + logscale);
  };

  global.Walks = W;
  if (typeof module !== "undefined" && module.exports) module.exports = W;
})(typeof window !== "undefined" ? window : globalThis);
