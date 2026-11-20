"""Эрдёш--Реньи: обход графа, пуассоновский мост и точная вероятность связности.

Три независимых инструмента:

    bfs_front()        -- точная симуляция процесса активных вершин
                          A_{t+1} = A_t - 1 + Bin(n - t - A_t, p);
    poisson_bridge()   -- выборка точного пуассоновского моста из леммы 2.1:
                          j ~ Multinomial(n-1, lambda/sum lambda);
    connectivity()     -- точная (не асимптотическая) вероятность связности
                          P_n(p) = (1-(1-p)^{n-1})^{n-1} * P(экскурсия),
                          где P(экскурсия) считается ДП по пуассоновским весам.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bfs_front", "poisson_lambda", "poisson_bridge",
           "excursion_probability", "connectivity", "connectivity_asymptotic",
           "giant_fraction"]


# ------------------------------------------------------------- обход графа

def bfs_front(n: int, p: float, rng: np.random.Generator | None = None,
              max_steps: int | None = None) -> dict:
    """Один прогон обхода: возвращает фронт A_t, t = 0..T.

    A_0 = 0 соответствует началу до обнаружения первой вершины; на шаге t
    обрабатывается одна активная вершина и открывается Bin(U_t, p) новых.
    Возвращаемое A_t -- число активных вершин после t шагов.
    """
    if rng is None:
        rng = np.random.default_rng()
    if max_steps is None:
        max_steps = 4 * n + 100
    a = 1                      # активных после обнаружения стартовой вершины
    u = n - 1                  # необнаруженных
    front = [a]
    t = 0
    while a > 0 and t < max_steps:
        x = int(rng.binomial(u, p)) if u > 0 else 0
        a = a - 1 + x
        u -= x
        front.append(a)
        t += 1
    front = np.asarray(front)
    connected = bool(front[-1] == 0 and u == 0)
    return dict(front=front, connected=connected, discovered=n - u,
                largest=u == 0, steps=len(front) - 1)


def giant_fraction(n: int, p: float, rng: np.random.Generator | None = None,
                   trials: int = 40) -> float:
    """Средняя доля вершин в компоненте стартовой вершины (оценка)."""
    if rng is None:
        rng = np.random.default_rng()
    vals = [bfs_front(n, p, rng)["discovered"] / n for _ in range(trials)]
    return float(np.mean(vals))


# ------------------------------------------------------- пуассоновский мост

def poisson_lambda(n: int, p: float) -> np.ndarray:
    """lambda_t = q (1-p)^{t-1}, t = 1..n-1, из леммы 2.1."""
    q = (n - 1) * p / (1.0 - (1.0 - p) ** (n - 1))
    t = np.arange(1, n)
    return q * (1.0 - p) ** (t - 1)


def poisson_bridge(n: int, p: float, size: int = 1,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Выборки моста S_k = sum_{i<=k}(X_i - 1), sum X_i = n-1.

    Точный способ: при условии на сумму независимые пуассоновские X_i
    распределены полиномиально, поэтому достаточно rng.multinomial.
    Возвращает массив (size, n) высот S_0..S_{n-1} (S_0 = 0, S_{n-1} = 0).
    """
    if rng is None:
        rng = np.random.default_rng()
    lam = poisson_lambda(n, p)
    probs = lam / lam.sum()
    j = rng.multinomial(n - 1, probs, size=size)
    steps = j - 1
    out = np.zeros((size, n + 1), dtype=int)
    out[:, 1:n] = np.cumsum(steps, axis=1)
    return out


# ------------------------------------------------- точная связность

def _poisson_weight_dp(n: int, p: float, positive: bool):
    """ДП по весам w_t(j) = lambda_t^j / j! с ограничением на знак пути.

    Путь: h_0 = 0, h_t = h_{t-1} + j_t - 1, t = 1..n-1, sum j_t = n-1.
    Нужно sum по путям с h_t >= 0 (если positive) веса prod w_t(j_t).
    Строки нормируются на максимум, логарифм нормировки накапливается,
    чтобы (n-1)! не переполняло float64.
    """
    lam = poisson_lambda(n, p)
    m = n - 1
    jmax = m
    off = m + 2
    width = 2 * m + 5
    # веса w_t(j) для всех t; логарифмы на будущее не нужны, но масштаб нужен
    w = np.empty((m, jmax + 1))
    logw = np.empty((m, jmax + 1))
    from math import lgamma
    for t in range(m):
        j = np.arange(jmax + 1)
        logw[t] = j * np.log(lam[t]) - np.array([lgamma(x + 1) for x in j])
        # общий сдвиг на строку не меняет отношений; убираем переполнение
        logw[t] -= logw[t].max()
        w[t] = np.exp(logw[t])
    row = np.zeros(width)
    row[off] = 1.0
    logscale = 0.0
    allowed = np.arange(width) >= off if positive else np.ones(width, dtype=bool)
    for t in range(m):
        new = np.zeros(width)
        wt = w[t]
        for j in range(jmax + 1):
            if wt[j] < 1e-300:
                continue
            shift = j - 1
            if shift >= 0:
                new[shift:] += wt[j] * row[:width - shift]
            else:
                new[:width + shift] += wt[j] * row[-shift:]
        new *= allowed
        mx = new.max()
        if mx <= 0:
            return -np.inf
        new /= mx
        logscale += np.log(mx)
        row = new
    # в конце h_{m} = 0, т.е. индекс off
    val = row[off]
    if val <= 0:
        return -np.inf
    return np.log(val) + logscale


def excursion_probability(n: int, p: float) -> float:
    """P(S_k >= 0, 1 <= k <= n-2 | sum X_i = n-1) -- точная вероятность."""
    lp = _poisson_weight_dp(n, p, True)
    la = _poisson_weight_dp(n, p, False)
    return float(np.exp(lp - la))


def connectivity(n: int, p: float) -> float:
    """Точная вероятность связности G(n,p).

    Формула: P_n(p) = (1-(1-p)^{n-1})^{n-1} * P(экскурсия), см. лемму 2.1.
    """
    return float((1.0 - (1.0 - p) ** (n - 1)) ** (n - 1)
                 * excursion_probability(n, p))


def connectivity_asymptotic(n: int, p: float) -> float:
    """Асимптотика P_n ~ (1-(1-p)^{n-1})^{n-1} (режим lambda_n -> oo)."""
    return float((1.0 - (1.0 - p) ** (n - 1)) ** (n - 1))
