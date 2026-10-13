#!/usr/bin/env python3
"""ЗАМКНУТАЯ ФОРМА погашения слоистого сектора. Пересчёт пункта 4.

`bdcontrol.py` показал: вчерашняя фаза неверна (w(0) = -1 начислялся
НЕСВЯЗАННЫМ парам слой1-слой3). Правильная фаза столбца:

    ph(T) = -|T| + sum_{i: m_i >= 1} w(m_i),    m_i = |s_i ∩ T|

С ней сумма берётся СТРУКТУРНО, без перебора 2^{a1 a2} матриц `M12`.

КАК. Пусть u = e^{-i theta}, theta = k_4/hbar, и
phi(m) = e^{i theta w(m)} для m >= 1, phi(0) = 1. Тогда
psi(m) = phi(m) - 1 отлична от нуля ТОЛЬКО при m = 1,2,3, потому что
w(m >= 4) = 0. Раскрываем произведение по строкам:

    G = sum_T u^{|T|} prod_i phi(m_i)
      = sum_{S ⊆ строки} sum_T u^{|T|} prod_{i in S} psi(m_i)

Второй слой распадается на классы по образцу вхождения в строки
(для a1 строк — 2^{a1} классов размеров n_c). В члене с данным `S`:
  - классы, не пересекающие `S`, суммируются СВОБОДНО: (1+u)^{их сумма};
  - классы, пересекающие `S`, обязаны иметь t_c <= 3, потому что
    m_i <= 3 для i in S. Их перебор КОНЕЧЕН и не зависит от a_2.

Значит G(n_c) считается за O(1) по a_2, и максимум по `M12` берётся
перебором КОМПОЗИЦИЙ a_2 по классам, а не матриц. Для a1 = 2 это
O(a_2^3) вариантов — то есть a_2 можно гнать до сотен.

Сверх того для a1 = 2 берётся и ПОЛНОЕ Z: число матриц `M12` с данными
размерами классов есть мультиномиальный коэффициент, поэтому

    Z = e^{i theta N} sum_{композиции} [a_2!/prod n_c!] u^{|M12|} G^{a_3}

то есть погашение по второму слою тоже учтено, а не оценено сверху.

КОНТРОЛИ:
  1. замкнутая форма против прямой суммы по 2^{a_2} подмножествам;
  2. полное Z против перечисления ВСЕХ порядков сектора с действием,
     посчитанным от матрицы отношений (`bdcontrol.bd_from_poset`);
  3. пустая M12: G = (1+u)^{a_2}, значит q = |cos(theta/2)|^{a_2};
  4. hbar -> oo: q -> 1.

  python3 cancelclosed.py
"""
import itertools
import math
from functools import lru_cache

import numpy as np

from bdcontrol import bd_from_poset

K4 = 4.0 / math.sqrt(6.0)
W = {1: 9.0, 2: -16.0, 3: 8.0}
MMAX = 3


def psi(m, th):
    """phi(m) - 1; ноль при m = 0 и при m >= 4."""
    if m <= 0 or m > MMAX:
        return 0.0 + 0.0j
    return np.exp(1j * th * W[m]) - 1.0


def comb(n, k):
    if k < 0 or k > n:
        return 0.0
    return float(math.comb(int(n), int(k)))


def g_closed(a1, nc, th):
    """G по размерам классов nc[c], c -- битовая маска по строкам."""
    u = np.exp(-1j * th)
    rows = range(a1)
    tot = 0.0 + 0.0j
    for S in range(1 << a1):
        Sset = [i for i in rows if (S >> i) & 1]
        free = 0
        con = []
        for c in range(1 << a1):
            if nc[c] == 0:
                continue
            if c & S:
                con.append(c)
            else:
                free += nc[c]
        base = (1.0 + u) ** free
        if S == 0:
            tot += base
            continue
        # перебор t_c для пересекающих классов при m_i <= 3, i in S
        acc = 0.0 + 0.0j

        def dfs(idx, tvec, msum):
            nonlocal acc
            if idx == len(con):
                term = 1.0 + 0.0j
                for i in Sset:
                    term *= psi(msum[i], th)
                if term == 0:
                    return
                for c, t in zip(con, tvec):
                    term *= comb(nc[c], t) * u ** t
                acc += term
                return
            c = con[idx]
            lim = min(MMAX, nc[c])
            for t in range(lim + 1):
                ms = list(msum)
                ok = True
                for i in Sset:
                    if (c >> i) & 1:
                        ms[i] += t
                        if ms[i] > MMAX:
                            ok = False
                            break
                if ok:
                    dfs(idx + 1, tvec + [t], ms)

        dfs(0, [], [0] * a1)
        tot += base * acc
    return tot


def g_brute(a1, nc, th, a2=None):
    """КОНТРОЛЬ 1: прямая сумма по подмножествам второго слоя."""
    # построить строки из классов
    pos = 0
    rows = [0] * a1
    for c in range(1 << a1):
        for _ in range(nc[c]):
            for i in range(a1):
                if (c >> i) & 1:
                    rows[i] |= 1 << pos
            pos += 1
    a2 = pos
    u = np.exp(-1j * th)
    tot = 0.0 + 0.0j
    for T in range(1 << a2):
        ph = 0.0
        for s in rows:
            m = bin(s & T).count('1')
            if 1 <= m <= MMAX:
                ph += W[m]
        tot += u ** bin(T).count('1') * np.exp(1j * th * ph)
    return tot, rows


def z_closed_a1_2(a2, a3, th):
    """Полное Z для a1 = 2 через композиции a_2 по четырём классам."""
    u = np.exp(-1j * th)
    tot = 0.0 + 0.0j
    lf = [math.lgamma(k + 1) for k in range(a2 + 1)]
    for n11 in range(a2 + 1):
        for n10 in range(a2 - n11 + 1):
            for n01 in range(a2 - n11 - n10 + 1):
                n00 = a2 - n11 - n10 - n01
                nc = {0: n00, 1: n10, 2: n01, 3: n11}
                g = g_closed(2, nc, th)
                lmult = lf[a2] - lf[n00] - lf[n10] - lf[n01] - lf[n11]
                m12 = n10 + n01 + 2 * n11
                tot += np.exp(lmult) * u ** m12 * g ** a3
    return tot


def z_poset(a1, a2, a3, th):
    """КОНТРОЛЬ 2: Z перечислением всех порядков, действие от матрицы."""
    tot = 0.0 + 0.0j
    for m12 in range(1 << (a1 * a2)):
        rows = [(m12 >> (i * a2)) & ((1 << a2) - 1) for i in range(a1)]
        for m23 in range(1 << (a2 * a3)):
            cols = [(m23 >> (k * a2)) & ((1 << a2) - 1) for k in range(a3)]
            S, _ = bd_from_poset(a1, a2, a3, rows, cols)
            tot += np.exp(1j * th * S)
    return tot


def maxq(a1, a2, th):
    """max_{M12} |G|/2^{a_2} перебором композиций (не матриц)."""
    best, arg = -1.0, None
    ncl = 1 << a1
    for comp in itertools.combinations(range(a2 + ncl - 1), ncl - 1):
        prev, parts = -1, []
        for c in comp:
            parts.append(c - prev - 1)
            prev = c
        parts.append(a2 + ncl - 2 - prev)
        nc = {c: parts[c] for c in range(ncl)}
        g = abs(g_closed(a1, nc, th)) / 2.0 ** a2
        if g > best:
            best, arg = g, dict(nc)
    return best, arg


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    th = K4 / 0.55
    print('КОНТРОЛЬ 1: замкнутая форма против прямой суммы по 2^{a_2}')
    rs = np.random.default_rng(7)
    worst = 0.0
    for a1 in (1, 2, 3):
        for _ in range(6):
            ncl = 1 << a1
            nc = {c: int(rs.integers(0, 4)) for c in range(ncl)}
            if sum(nc.values()) == 0:
                continue
            gc = g_closed(a1, nc, th)
            gb, rows = g_brute(a1, nc, th)
            rel = abs(gc - gb) / max(abs(gb), 1e-12)
            worst = max(worst, rel)
            print(f'  a1={a1} классы={list(nc.values())} a2={sum(nc.values())}'
                  f': |G| замкн {abs(gc):.8f}, прямо {abs(gb):.8f}, '
                  f'отн. {rel:.2e}')
    print(f'  худшее расхождение {worst:.2e}')
    print()
    print('КОНТРОЛЬ 2: полное Z (a1=2) против перечисления ПОРЯДКОВ')
    for (a2, a3) in ((2, 2), (3, 2), (2, 3)):
        zc = z_closed_a1_2(a2, a3, th)
        zp = z_poset(2, a2, a3, th)
        zc = zc * np.exp(1j * th * (2 + a2 + a3))
        print(f'  слои (2,{a2},{a3}): |Z| замкн {abs(zc):.8f}, '
              f'перечисление {abs(zp):.8f}, отн. '
              f'{abs(zc-zp)/max(abs(zp),1e-12):.2e}')
    print()
    print('КОНТРОЛЬ 3: пустая M12 -> q = |cos(theta/2)|^{a_2}')
    for a2 in (4, 8, 16):
        nc = {0: a2, 1: 0, 2: 0, 3: 0}
        q = abs(g_closed(2, nc, th)) / 2.0 ** a2
        pred = abs(math.cos(th / 2)) ** a2
        print(f'  a2={a2:>3}: q = {q:.6e}, |cos(theta/2)|^a2 = {pred:.6e}')
    print()
    print('КОНТРОЛЬ 4: hbar -> oo, подавления быть не должно')
    for hb in (0.55, 5.0, 50.0, 500.0):
        t2 = K4 / hb
        nc = {0: 1, 1: 2, 2: 2, 3: 2}
        q = abs(g_closed(2, nc, t2)) / 2.0 ** 7
        print(f'  hbar = {hb:>6}: q = {q:.6f}')


if __name__ == '__main__':
    main()
