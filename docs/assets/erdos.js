/* Эрдёш--Реньи: обход графа, пуассоновский мост, точная связность.
 *
 * Прямой аналог viz/common/erdos.py.  Формулы:
 *   фронт  A_{t+1} = A_t − 1 + Bin(n − t − A_t, p),
 *   мост   λ_t = q(1−p)^{t−1},  X ~ Multinomial(n−1, λ/Σλ),
 *   связность P_n(p) = (1 − (1−p)^{n−1})^{n−1} · P(экскурсия).
 */
(function (global) {
  "use strict";

  var E = {};

  /* Один прогон обхода: A_0 = 1 (стартовая вершина уже обнаружена). */
  E.bfsFront = function (n, p, rng) {
    var a = 1, u = n - 1, t = 0;
    var front = [1];
    var maxSteps = 4 * n + 100;
    while (a > 0 && t < maxSteps) {
      var x = u > 0 ? rng.binomial(u, p) : 0;
      a = a - 1 + x;
      u -= x;
      front.push(a);
      t++;
    }
    return {
      front: front,
      connected: front[front.length - 1] === 0 && u === 0,
      discovered: n - u,
      steps: front.length - 1
    };
  };

  E.poissonLambda = function (n, p) {
    var denom = 1 - Math.pow(1 - p, n - 1);
    var q = denom > 0 ? (n - 1) * p / denom : n - 1;
    var lam = new Float64Array(n - 1);
    for (var t = 1; t <= n - 1; t++) lam[t - 1] = q * Math.pow(1 - p, t - 1);
    return lam;
  };

  /* Точные выборки пуассоновского моста: высоты S_0..S_n, S_0 = S_n = 0. */
  E.poissonBridge = function (n, p, size, rng) {
    var lam = E.poissonLambda(n, p);
    var total = 0, i;
    for (i = 0; i < lam.length; i++) total += lam[i];
    var probs = new Float64Array(lam.length);
    for (i = 0; i < lam.length; i++) probs[i] = lam[i] / total;

    var out = [];
    for (var s = 0; s < size; s++) {
      var j = rng.multinomial(n - 1, probs);
      var path = new Int32Array(n + 1);
      var acc = 0;
      for (i = 0; i < n - 1; i++) {
        acc += j[i] - 1;
        path[i + 1] = acc;
      }
      out.push(path);
    }
    return out;
  };

  /* Логарифм суммы весов путей h_1..h_{n-1} с h_t >= 0 (positive=true)
     или без ограничения; веса w_t(j) = λ_t^j / j!. */
  function poissonWeightLog(n, p, positive) {
    var lam = E.poissonLambda(n, p);
    var m = n - 1, off = m + 2, width = 2 * m + 5;
    var row = new Float64Array(width);
    row[off] = 1.0;
    var logscale = 0.0;

    for (var t = 0; t < m; t++) {
      /* w[j] = λ^j / j!, обрываем хвост на 1e-18 */
      var w = new Float64Array(m + 1);
      w[0] = 1.0;
      var jmax = 0;
      for (var j = 1; j <= m; j++) {
        w[j] = w[j - 1] * lam[t] / j;
        if (w[j] > 1e-18) jmax = j;
      }
      var next = new Float64Array(width);
      for (var jj = 0; jj <= jmax; jj++) {
        var wt = w[jj];
        if (!(wt > 0)) continue;
        var shift = jj - 1;
        if (shift >= 0) {
          for (var idx = shift; idx < width; idx++) next[idx] += wt * row[idx - shift];
        } else {
          for (var idx2 = 0; idx2 < width + shift; idx2++) next[idx2] += wt * row[idx2 - shift];
        }
      }
      if (positive) {
        for (var z = 0; z < off; z++) next[z] = 0;
      }
      var mx = 0;
      for (var q = 0; q < width; q++) if (next[q] > mx) mx = next[q];
      if (!(mx > 0)) return -Infinity;
      for (var q2 = 0; q2 < width; q2++) next[q2] /= mx;
      logscale += Math.log(mx);
      row = next;
    }
    var val = row[off];
    return val > 0 ? Math.log(val) + logscale : -Infinity;
  }

  E.excursionProbability = function (n, p) {
    var lp = poissonWeightLog(n, p, true);
    var la = poissonWeightLog(n, p, false);
    if (!isFinite(lp) || !isFinite(la)) return 0;
    return Math.exp(lp - la);
  };

  E.connectivity = function (n, p) {
    var a = 1 - Math.pow(1 - p, n - 1);
    var factor = Math.pow(a, n - 1);
    return factor * E.excursionProbability(n, p);
  };

  E.connectivityAsymptotic = function (n, p) {
    return Math.pow(1 - Math.pow(1 - p, n - 1), n - 1);
  };

  global.Erdos = E;
  if (typeof module !== "undefined" && module.exports) module.exports = E;
})(typeof window !== "undefined" ? window : globalThis);
