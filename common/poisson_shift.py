"""Блуждание с шагами X_i = Y_i - 2, Y_i ~ Poisson(λ), и большие уклонения.

Модель.  X_1,...,X_n независимы, X_i = Y_i - 2, Y_i ~ Poisson(λ_i),
S_0 = 0, S_i = X_1 + ... + X_i.  При λ = 1 снос равен -1, дисперсия 1.
Шаг принимает значения -2, -1, 0, 1, ...; хвост пуассоновского шага
обрезан там, где его масса < 10^{-16}.

Конец в зоне больших уклонений: k = θ n, θ > 0.  Преобразование Крамера
переводит шаг снова в пуассоновский:

    E e^{αX} = exp(λ(e^α - 1) - 2α),   X^{(α)} ~ Poisson(λ e^α) - 2,

поэтому при постоянном λ = 1 наклонённый шаг имеет λ' = θ + 2 и снос θ.
Условный закон P(. | S_n = k) в точности совпадает с P_θ(. | S_n = k),
так как отношение правдоподобия e^{αk} постоянно на событии {S_n = k}.
Более того, при фиксированном k/n закон моста не зависит от λ: отношение
правдоподобий двух пуассоновских моделей тоже постоянно на {S_n = k}.

Главный объект: τ = последний момент i < n, когда S_i < 0.  Для
наклонённого блуждания с положительным сносом «поздний» выход ниже нуля
требует большого уклонения влево, поэтому P(τ >= t | S_n = k) убывает
экспоненциально по t.  Функции ниже считают это точно, без Монте-Карло:
образец моста нужен только для картинки, а сами вероятности берутся из ДП.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

__all__ = ["LAM", "SHIFT", "kernel", "log_kernel", "poisson_pmf",
           "sample_bridge", "tau_of_paths", "tau_survival"]

LAM = 1.0
SHIFT = 2


def kernel(lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Распределение шага X = Y - 2: пары (d, w), d = -2, -1, 0, ..."""
    dmax = int(math.ceil(lam + 8.0 * math.sqrt(lam) + 12.0))
    j = np.arange(dmax + 1)
    w = np.exp(-lam + j * math.log(lam) - gammaln(j + 1))
    return j - SHIFT, w


def log_kernel(lam: float) -> tuple[np.ndarray, np.ndarray]:
    """То же в логарифмах (нужно для последовательной выборки моста)."""
    dmax = int(math.ceil(lam + 8.0 * math.sqrt(lam) + 12.0))
    j = np.arange(dmax + 1)
    return j - SHIFT, -lam + j * math.log(lam) - gammaln(j + 1)


def poisson_pmf(lam: float, m: int) -> float:
    """P(Poisson(lam) = m)."""
    if m < 0:
        return 0.0
    if lam <= 0.0:
        return 1.0 if m == 0 else 0.0
    return math.exp(-lam + m * math.log(lam) - math.lgamma(m + 1))


def _log_pois_arr(lam: float, m: np.ndarray) -> np.ndarray:
    """log P(Poisson(lam) = m) поэлементно; вне носителя -inf."""
    out = np.full(m.shape, -np.inf)
    ok = m >= 0
    if lam > 0.0:
        out[ok] = -lam + m[ok] * math.log(lam) - gammaln(m[ok] + 1)
    else:
        out[ok] = np.where(m[ok] == 0, 0.0, -np.inf)
    return out


def sample_bridge(n: int, theta: float, m: int,
                  rng: np.random.Generator, jmax: int | None = None):
    """Точные выборки моста S_n = k, k = round(θ n), m штук.

    Выборка идёт в наклонённой мере (она же условная) последовательными
    весами p_θ(X_t = d) P_θ(S_{n-t-1} = k - x - d).  Возвращает массив
    (m, n+1) целых высот S_0..S_n.
    """
    lam = theta + SHIFT
    d, logw = log_kernel(lam)
    if jmax is not None:
        d, logw = d[:jmax + 1], logw[:jmax + 1]
    k = int(round(theta * n))
    paths = np.zeros((m, n + 1), dtype=np.int32)
    x = np.zeros(m, dtype=np.int64)
    lw = np.empty((m, d.size))
    for t in range(n):
        rem = n - t - 1
        need = (k - x)[:, None] - d[None, :]           # требуемая сумма остатка
        lw[:] = logw[None, :] + _log_pois_arr(
            rem * lam, need + SHIFT * rem)
        lw -= lw.max(axis=1, keepdims=True)
        cw = np.cumsum(np.exp(lw), axis=1)
        u = rng.random(m) * cw[:, -1]
        idx = np.minimum((cw < u[:, None]).sum(axis=1), d.size - 1)
        x = x + d[idx]
        paths[:, t + 1] = x
    return paths


def tau_of_paths(paths) -> np.ndarray:
    """τ = последний i с S_i < 0 (0, если выхода не было)."""
    paths = np.asarray(paths)
    idx = np.arange(paths.shape[1])[None, :]
    return np.where(paths < 0, idx, 0).max(axis=1)


def tau_survival(n: int, theta: float, T: int) -> np.ndarray:
    """Точная P(τ >= t | S_n = k) для t = 0..T, k = round(θ n).

    Обратная динамика: G_m(x) = P_x(S_1..S_{m-1} >= 0, S_m = k) для
    наклонённого блуждания, затем
        P(τ >= t) = 1 - Σ_{x>=0} P_θ(S_t = x) G_{n-t}(x) / P_θ(S_n = k).
    Для t >= 1 это вероятность того, что последний выход ниже нуля
    случился не раньше t; out[0] = 1.
    """
    lam = theta + SHIFT
    d, w = kernel(lam)
    k = int(round(theta * n))
    H = k + SHIFT * n + 10
    rows: list[np.ndarray] = [np.zeros(H + 1) for _ in range(min(n, T) + 1)]
    prev = np.zeros(H + 1)
    prev[k] = 1.0
    rows[0][:] = prev
    keep_from = max(0, n - T)
    for m in range(1, n + 1):
        cur = np.zeros(H + 1)
        for dd, ww in zip(d, w):
            s = int(dd)
            if s >= 0:
                cur[:H + 1 - s] += ww * prev[s:]
            else:
                s = -s
                cur[s:] += ww * prev[:H + 1 - s]
        prev = cur
        if m >= keep_from:
            rows[m - keep_from][:] = cur

    xmin = -SHIFT * T - 1
    grid = np.arange(xmin, H + 1)
    f = np.zeros(grid.size)
    f[0 - xmin] = 1.0
    den = poisson_pmf(n * lam, k + SHIFT * n)
    out = np.zeros(T + 1)
    for t in range(T + 1):
        if t > n:
            break
        if t > 0:
            nf = np.zeros_like(f)
            for dd, ww in zip(d, w):
                s = int(dd)
                if s >= 0:
                    nf[s:] += ww * f[:f.size - s]
                else:
                    s = -s
                    nf[:f.size - s] += ww * f[s:]
            f = nf
        g = rows[(n - t) - keep_from]
        joint = 0.0
        ok = grid >= 0
        joint = float(f[ok] @ g[grid[ok]])
        out[t] = 1.0 - joint / den
    out[0] = 1.0
    return np.clip(out, 0.0, 1.0)


if __name__ == "__main__":
    import sys

    n, theta = 400, 1.0
    p = sample_bridge(n, theta, 2000, np.random.default_rng(7))
    tau = tau_of_paths(p)
    sv = tau_survival(n, theta, 20)
    print("выборок:", p.shape, "доля с выходом:", (tau > 0).mean().round(4))
    for t in (0, 1, 2, 3, 5, 8, 12, 20):
        print(f"t={t:2d}  Монте-Карло {(tau >= t).mean():.4f}  "
              f"точно {sv[t]:.4f}")
    print("квантили τ среди вышедших:",
          np.percentile(tau[tau > 0], [50, 90, 99, 100]))
    sys.exit(0)
