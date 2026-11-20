/* Детерминированный генератор случайных чисел (mulberry32).
 *
 * Нужен, чтобы при каждом открытии страницы и каждом движении ползунка
 * картинка была воспроизводимой: все демонстрации на сайте используют
 * только этот генератор, Math.random нигде не вызывается.
 */
(function (global) {
  "use strict";

  function RNG(seed) {
    this.s = (seed >>> 0) || 0x9e3779b9;
  }

  /* Равномерное число на [0, 1). */
  RNG.prototype.next = function () {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    var t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  RNG.prototype.uniform = function (a, b) {
    return a + (b - a) * this.next();
  };

  RNG.prototype.int = function (n) {
    return Math.floor(this.next() * n);
  };

  /* Стандартная нормаль (Бокс--Мюллер). */
  RNG.prototype.normal = function () {
    var u = 1 - this.next();
    var v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  /* Биномиальное число.  Точный цикл при малом n·p, иначе нормальная
     аппроксимация — для картинки этого достаточно. */
  RNG.prototype.binomial = function (n, p) {
    if (n <= 0 || p <= 0) return 0;
    if (p >= 1) return n;
    var mean = n * p;
    if (n < 80 || mean < 20 || n * (1 - p) < 20) {
      var x = 0;
      for (var i = 0; i < n; i++) if (this.next() < p) x++;
      return x;
    }
    var sd = Math.sqrt(n * p * (1 - p));
    var v = Math.round(mean + sd * this.normal());
    return v < 0 ? 0 : (v > n ? n : v);
  };

  /* Полиномиальная выборка через цепочку биномиальных. */
  RNG.prototype.multinomial = function (m, probs) {
    var k = probs.length;
    var out = new Int32Array(k);
    var rest = 1.0;
    var rem = m;
    for (var i = 0; i < k - 1; i++) {
      if (rem <= 0 || rest <= 0) break;
      var pi = Math.min(1, probs[i] / rest);
      var x = this.binomial(rem, pi);
      out[i] = x;
      rem -= x;
      rest -= probs[i];
    }
    out[k - 1] = Math.max(0, rem);
    return out;
  };

  global.RNG = RNG;
  if (typeof module !== "undefined" && module.exports) module.exports = RNG;
})(typeof window !== "undefined" ? window : globalThis);
