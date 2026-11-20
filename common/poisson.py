"""Точные вычисления для блуждания с шагами X_i = Y_i - 1, Y_i ~ Poisson(λ_i).

Модель.  X_1,...,X_n независимы, X_i = Y_i - 1, Y_i ~ Poisson(λ_i).
S_0 = 0, S_i = X_1 + ... + X_i.  Шаг принимает значения -1, 0, 1, 2, ...
Поэтому мост S_n = k и положительность считаются динамикой по (i, h) с
усечением пуассоновского хвоста шага (ошибка < 10^{-15}) и окна высот
(ошибка < 10^{-20}).

Преобразование Крамера для шага Пуассона устроено просто:
    E e^{θX} = exp(λ(e^θ - 1) - θ),
поэтому наклонённый шаг снова пуассоновский:
    λ^{(θ)} = λ e^θ,   E_θ X = λ e^θ - 1.
Условие E_θ X = k/n даёт λ' = 1 + k/n для постоянного λ (и вообще
λ'_i = λ_i (n+k)/Σλ).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["kernel", "tilt_scale", "tilted_profile", "bridge_curve",
           "positivity", "poisson_pmf"]


def kernel(lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Распределение шага X = Y - 1: пары (d, w), d = -1, 0, 1, ...

    Хвост обрезан там, где P(Y > d) < 10^{-16}.
    """
    dmax = int(math.ceil(lam + 8.0 * math.sqrt(lam) + 12.0))
    d = np.arange(-1, dmax + 1)
    w = np.empty(dmax + 2)
    w[0] = math.exp(-lam)                       # P(Y = 0)  ->  X = -1
    for j in range(dmax + 1):                   # w[j+1] = P(Y = j+1)
        w[j + 1] = w[j] * lam / (j + 1)
    return d, w


def poisson_pmf(lam: float, m: int) -> float:
    """P(Poisson(lam) = m) в логарифмическом масштабе."""
    if m < 0:
        return 0.0
    if lam <= 0.0:
        return 1.0 if m == 0 else 0.0
    return math.exp(-lam + m * math.log(lam) - math.lgamma(m + 1))


def tilt_scale(lam, n: int, k: int) -> float:
    """Множитель e^θ, при котором E_θ S_n = k: (n+k)/Σλ_i."""
    lam = np.asarray(lam, dtype=float)
    return (n + k) / float(lam.sum())


def tilted_profile(lam, n: int, k: int) -> np.ndarray:
    """Профиль λ^{(θ)} = λ_i e^θ, дающий E S_n = k."""
    lam = np.asarray(lam, dtype=float)
    return lam * tilt_scale(lam, n, k)


def _convolve(prev: np.ndarray, d: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Свёртка строки с распределением шага."""
    H = prev.size
    cur = np.zeros(H)
    for dd, ww in zip(d, w):
        s = int(dd)
        if s >= 0:
            if s < H:
                cur[s:] += ww * prev[:H - s]
        else:
            s = -s
            if s < H:
                cur[:H - s] += ww * prev[s:]
    return cur


def bridge_curve(n: int, lam: float, r: int = 0,
                 kmax: int | None = None) -> np.ndarray:
    """P(S_i > r, 1 <= i < n | S_n = k) для всех k <= kmax.

    Один прямой проход ДП с поглощающим барьером h <= r даёт совместные
    вероятности P(выжить, S_n = k); делим на P(Poisson(nλ) = n+k).
    Возвращает массив длины kmax+1 (индекс -- k; для k < 0 значение 0).
    """
    if kmax is None:
        kmax = int(round(1.25 * n)) + 4
    d, w = kernel(lam)
    hi = kmax + n                    # выше не вернуться к k <= kmax
    lo = min(0, r)
    off = -lo
    H = hi - lo + 1
    prev = np.zeros(H)
    prev[off] = 1.0
    logscale = 0.0
    for i in range(1, n + 1):
        cur = _convolve(prev, d, w)
        if i < n:
            cur[:max(r + off + 1, 0)] = 0.0
        mx = cur.max()
        if mx <= 0.0:
            return np.zeros(kmax + 1)
        cur /= mx
        logscale += math.log(mx)
        prev = cur
    num = prev * math.exp(logscale)
    out = np.zeros(kmax + 1)
    for k in range(kmax + 1):
        m = n + k
        den = poisson_pmf(n * lam, m)
        if den > 0.0:
            out[k] = num[k + off] / den
    return np.clip(out, 0.0, 1.0)


def positivity(lam, r: int = 0, sigmas: float = 10.0) -> float:
    """P(S_i > r, 1 <= i < n) для профиля λ_i, точно с точностью до хвоста."""
    lam = np.asarray(lam, dtype=float)
    n = lam.size
    mu = np.cumsum(lam - 1.0)
    var = np.cumsum(lam)
    hi_max = int(max(8, np.max(np.ceil(mu + sigmas * np.sqrt(var) + 20.0))))
    lo = min(0, r)
    off = -lo
    prev = np.zeros(hi_max - lo + 1)
    prev[off] = 1.0
    logscale = 0.0
    for i in range(1, n + 1):
        d, w = kernel(float(lam[i - 1]))
        cur = _convolve(prev, d, w)
        if i < n:
            cur[:max(r + off + 1, 0)] = 0.0
        hi = int(math.ceil(mu[i - 1] + sigmas * math.sqrt(var[i - 1]) + 20.0))
        if hi + off + 1 < cur.size:
            cur[hi + off + 1:] = 0.0     # отброшенный верхний хвост
        mx = cur.max()
        if mx <= 0.0:
            return 0.0
        cur /= mx
        logscale += math.log(mx)
        prev = cur
    return float(min(1.0, prev.sum() * math.exp(logscale)))
