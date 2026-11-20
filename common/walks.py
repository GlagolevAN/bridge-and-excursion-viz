"""Точные вычисления для неоднородного блуждания с шагами ±1.

Модель.  X_1,...,X_n независимы, X_i ∈ {+1,-1}, P(X_i = +1) = p_i.
S_0 = 0, S_i = X_1 + ... + X_i.  Матрица переходов на каждом шаге своя,
поэтому «мост» (условие S_n = k) и условие положительности считаются
точно: динамическим программированием по (номер шага, высота) без
сэмплирования с отвержением.

Основные объекты:

    Bridge       -- точное условное распределение моста S_n = k
                    (при желании -- с запретом S_i <= r, 1 <= i < n);
    positivity() -- точная вероятность P(S_i > r, 1 <= i < n | S_n = k);
    sample()     -- независимые точные выборки из этого условного закона.

Все дроби считаются в логарифмическом масштабе (строчная нормировка),
поэтому большие n не приводят к переполнению.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Bridge", "const_profile", "linear_drift_profile",
           "two_block_profile", "sublinear_profile", "parity_k"]


def parity_k(n: int, target: float) -> int:
    """Ближайшее к target целое k той же чётности, что и n (для шагов ±1)."""
    k = int(round(target))
    if (n - k) % 2 != 0:
        k -= 1
    return k


# ---------------------------------------------------------------- профили p_i

def const_profile(n: int, drift: float) -> np.ndarray:
    """Однородное блуждание: E X_i = drift (|drift| < 1)."""
    assert abs(drift) < 1.0
    return np.full(n, 0.5 * (1.0 + drift))


def linear_drift_profile(n: int, c: float) -> np.ndarray:
    """Условие теоремы 1: mu_{n,i} = c*i, т.е. P(X_i=+1) = (1+c)/2.

    Снос на каждом шаге постоянен, но накопленное среднее растёт линейно.
    """
    return const_profile(n, c)


def sublinear_profile(n: int, c: float, alpha: float) -> np.ndarray:
    """Условие теоремы о сублинейном сносе: mu_{n,i} = c*i*n^{alpha-1}.

    Эквивалентно пошаговому сносу c*n^{alpha-1} -> 0 при alpha < 1.
    """
    drift = c * n ** (alpha - 1.0)
    drift = min(drift, 0.98)
    return const_profile(n, drift)


def two_block_profile(n: int, c: float, m: int | None = None) -> np.ndarray:
    """Профиль из теоремы об экскурсии: снос +c до m и -c после m.

    mu_{n,i} = c*i при i <= m и mu_{n,i} = -c*(n-i) при i > m; для шагов ±1
    это отвечает пошаговому сносу +c на первой половине и -c на второй,
    так что типичная траектория -- подъём и спуск (экскурсия).
    """
    if m is None:
        m = n // 2
    drift = np.empty(n)
    drift[:m] = c
    drift[m:] = -c
    return 0.5 * (1.0 + drift)


# ------------------------------------------------------------------- мост

class Bridge:
    """Точное условное распределение S_n = k для шагов ±1.

    Parameters
    ----------
    p : array_like, shape (n,)
        Вероятности шага +1.
    k : int
        Конечное состояние S_n.
    r : int or None
        Если задано, траектория обязана удовлетворять S_i > r при 1 <= i < n
        (и k > r, иначе условная вероятность равна нулю).
    """

    def __init__(self, p, k: int, r: int | None = None):
        p = np.asarray(p, dtype=float)
        self.p = p
        self.n = len(p)
        self.k = int(k)
        self.r = r
        # шаги ±1: S_n имеет ту же чётность, что и n
        if (self.n - self.k) % 2 != 0:
            raise ValueError(
                f"недостижимое конечное состояние: n={self.n}, k={self.k} "
                f"(n-k должно быть чётным)")
        self._rows: np.ndarray | None = None
        self._logpos: float = 0.0
        self._rows_all: np.ndarray | None = None
        self._logall: float = 0.0

    # --- динамика -------------------------------------------------------
    def _forward(self, restricted: bool, keep: bool = True):
        """Прямой проход ДП. Возвращает (строки или последнюю строку, logscale).

        Строка i -- распределение S_i с точностью до общего множителя;
        множители накапливаются в log-масштабе.  При keep=False хранится
        только последняя строка (память O(n) вместо O(n^2)).
        """
        n, r, k = self.n, self.r, self.k
        width = 2 * n + 3
        off = n + 1                     # индекс высоты h: h + off
        rows = np.zeros((n + 1, width)) if keep else np.zeros((2, width))
        rows[0, off] = 1.0
        logscale = 0.0
        if restricted and r is not None:
            # при 1 <= i < n разрешены только h > r
            interior = np.arange(width) > (off + r)
        else:
            interior = np.ones(width, dtype=bool)
        final = np.arange(width) == (off + k)
        cur = rows[0] if keep else rows[1]
        for i in range(1, n + 1):
            prev = rows[i - 1] if keep else rows[1 - (i % 2)]
            cur = rows[i] if keep else rows[i % 2]
            cur[:] = 0.0
            pi = self.p[i - 1]
            # h-1 -> h (шаг +1)  и  h+1 -> h (шаг -1)
            cur[1:] = pi * prev[:-1]
            cur[:-1] += (1.0 - pi) * prev[1:]
            cur *= interior if i < n else final
            mx = cur.max()
            if mx <= 0.0:
                raise ValueError(
                    f"условная вероятность равна нулю (n={n}, k={k}, r={r}, шаг {i})")
            cur /= mx
            logscale += np.log(mx)
        return rows, logscale

    def rows(self, restricted: bool = True):
        if restricted:
            if self._rows is None:
                self._rows, self._logpos = self._forward(True)
            return self._rows
        if self._rows_all is None:
            self._rows_all, self._logall = self._forward(False)
        return self._rows_all

    def _last(self, restricted: bool):
        """Последняя строка ДП без хранения всей истории (память O(n))."""
        rows, logscale = self._forward(restricted, keep=False)
        return rows[self.n % 2], logscale

    # --- вероятность ----------------------------------------------------
    def probability(self) -> float:
        """P(S_i > r, 1 <= i < n | S_n = k) -- точно."""
        if self.r is None:
            return 1.0
        # Концы не контролируются: S_0 = 0 и S_n = k могут быть <= r;
        # граница стоит только при 1 <= i < n.
        off = self.n + 1
        try:
            pos_row, logpos = self._last(True)
        except ValueError:
            # событие несовместимо с барьером (например, шаг ±1 и r >= 1)
            return 0.0
        all_row, logall = self._last(False)
        pos = pos_row[off + self.k]
        allp = all_row[off + self.k]
        if allp <= 0:
            return 0.0
        return float(np.exp(np.log(pos) - np.log(allp) + logpos - logall))

    # --- выборка --------------------------------------------------------
    def sample(self, size: int = 1, rng: np.random.Generator | None = None,
               restricted: bool = True) -> np.ndarray:
        """Независимые выборки моста; возвращает массив (size, n+1)."""
        if rng is None:
            rng = np.random.default_rng()
        rows = self.rows(restricted)
        n, off = self.n, self.n + 1
        h = np.full(size, self.k, dtype=int)
        paths = np.empty((size, n + 1), dtype=int)
        paths[:, n] = h
        for i in range(n, 0, -1):
            prev = rows[i - 1]
            pi = self.p[i - 1]
            # веса двух продолжений назад: из h в h-1 (шаг +1) и в h+1 (шаг -1)
            up = pi * prev[h - 1 + off]
            dn = (1.0 - pi) * prev[h + 1 + off]
            tot = up + dn
            u = rng.random(size)
            go_up = u * tot < up
            # если tot == 0 (не должно случаться), оставляем шаг вниз
            h = np.where(go_up, h - 1, h + 1)
            paths[:, i - 1] = h
        return paths

    # --- вспомогательное ------------------------------------------------
    def _posterior(self, restricted: bool = True) -> np.ndarray:
        """Поточечные маргиналы моста P(S_i = h | S_n = k).

        Прямые строки ДП сами по себе не conditioned на конец: в них ещё нет
        условия S_n = k (оно накладывается только на последнем шаге), поэтому
        их среднее — это среднее безусловного блуждания, а не моста.  Здесь
        прямой проход домножается на обратные веса B_i(h) = P(дойти из (i, h)
        в (n, k) с учётом запрета); произведение нормируется по h.
        """
        rows = self.rows(restricted)
        n, off, k, r = self.n, self.n + 1, self.k, self.r
        width = 2 * n + 3
        bwd = np.zeros((n + 1, width))
        bwd[n, off + k] = 1.0
        interior = np.arange(width) > (off + r) if restricted and r is not None else None
        for i in range(n - 1, -1, -1):
            pi, qi = self.p[i], 1.0 - self.p[i]
            nxt, cur = bwd[i + 1], bwd[i]
            cur[:-1] += pi * nxt[1:]         # h -> h+1 на шаге i+1
            cur[1:] += qi * nxt[:-1]         # h -> h-1 на шаге i+1
            if i >= 1 and interior is not None:
                cur *= interior
            mx = cur.max()
            if mx > 0.0:
                cur /= mx
        out = np.empty((n + 1, width))
        for i in range(n + 1):
            w = rows[i] * bwd[i]
            s = w.sum()
            out[i] = w / s if s > 0 else 0.0
        return out

    def mean_path(self, restricted: bool = True) -> np.ndarray:
        """E[S_i | условие] для всех i: условное среднее, а не прямое ДП."""
        n, off = self.n, self.n + 1
        idx = np.arange(2 * n + 3) - off
        post = self._posterior(restricted)
        out = np.empty(n + 1)
        for i in range(n + 1):
            row = post[i]
            s = row.sum()
            out[i] = (row * idx).sum() / s if s > 0 else np.nan
        return out

    def band(self, q: float = 0.9, restricted: bool = True) -> np.ndarray:
        """Поточечные квантили высоты моста: массив (n+1,) уровня q."""
        n, off = self.n, self.n + 1
        idx = np.arange(2 * n + 3) - off
        post = self._posterior(restricted)
        out = np.empty(n + 1)
        for i in range(n + 1):
            row = post[i]
            s = row.sum()
            if s <= 0:
                out[i] = np.nan
                continue
            cdf = np.cumsum(row) / s
            out[i] = idx[np.searchsorted(cdf, q)]
        return out


# ------------------------------------------------- вероятность без моста

def positivity_unconditioned(p, r: float = 0.0) -> np.ndarray:
    """P(S_j > r, 1 <= j <= i) для всех i -- точно, без условия на S_n.

    Возвращает массив длины n+1 (элемент 0 равен 1).
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    width = 2 * n + 3
    off = n + 1
    prev = np.zeros(width)
    cur = np.zeros(width)
    prev[off] = 1.0
    interior = np.arange(width) > (off + r)
    logscale = 0.0
    res = np.empty(n + 1)
    res[0] = 1.0
    for i in range(1, n + 1):
        cur[:] = 0.0
        pi = p[i - 1]
        cur[1:] = pi * prev[:-1]
        cur[:-1] += (1.0 - pi) * prev[1:]
        cur *= interior
        mx = cur.max()
        if mx <= 0.0:
            res[i:] = 0.0
            return res
        cur /= mx
        logscale += np.log(mx)
        # масса всех путей, остававшихся выше r до момента i
        res[i] = cur.sum() * np.exp(logscale)
        prev, cur = cur, prev
    return np.clip(res, 0.0, 1.0)
