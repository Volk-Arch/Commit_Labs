#!/usr/bin/env python3
"""ПРОВЕРКА НИЖНЕЙ ОЦЕНКИ и тонкий скан по hbar.

Вердикт «сектор выживает» опирается на НИЖНЮЮ оценку |Z|: вклад одного
структурного класса. Внутри класса все `M12` дают ОДНО И ТО ЖЕ |G|,
одно и то же |M12| и один и тот же arg G (перестановка метрк второго
слоя не меняет G), поэтому они складываются КОГЕРЕНТНО, и

    ln|Z| >~ ln(число M12 в классе) + a_3 ln|G(класс)|

Здесь это проверяется при a1 = 3 — где класс уже не тривиален — против
ПРЯМОГО перечисления всех 2^{3 a_2} матриц M12 с G, посчитанным суммой
по 2^{a_2} подмножествам (то есть без формулы вовсе).

  python3 gcrit2.py
"""
import itertools
import math

import numpy as np

from gcrit import g_md

K4 = 4.0 / math.sqrt(6.0)
WD = {1: 9.0, 2: -16.0, 3: 8.0}


def g_rows(rows, a2, th):
    """G прямой суммой по подмножествам, строки заданы явно."""
    u = np.exp(-1j * th)
    tot = 0.0 + 0.0j
    for T in range(1 << a2):
        ph = 0.0
        for s in rows:
            k = bin(s & T).count('1')
            if 1 <= k <= 3:
                ph += WD[k]
        tot += u ** bin(T).count('1') * np.exp(1j * th * ph)
    return tot


def z_direct(a1, a2, a3, th):
    """Z прямым перечислением M12, G -- суммой по подмножествам."""
    u = np.exp(-1j * th)
    tot = 0.0 + 0.0j
    for m12 in range(1 << (a1 * a2)):
        rows = [(m12 >> (i * a2)) & ((1 << a2) - 1) for i in range(a1)]
        pop = bin(m12).count('1')
        g = g_rows(rows, a2, th)
        tot += u ** pop * g ** a3
    return tot


def z_classes(a1, a2, a3, th):
    """Z суммой по композициям с мультиномиальными весами."""
    from cancelclosed import g_closed
    u = np.exp(-1j * th)
    ncl = 1 << a1
    lf = [math.lgamma(k + 1) for k in range(a2 + 1)]
    terms = []
    for comb in itertools.combinations(range(a2 + ncl - 1), ncl - 1):
        prev, parts = -1, []
        for c in comb:
            parts.append(c - prev - 1)
            prev = c
        parts.append(a2 + ncl - 2 - prev)
        nc = {c: parts[c] for c in range(ncl)}
        g = g_closed(a1, nc, th)
        m12 = sum(parts[c] * bin(c).count('1') for c in range(ncl))
        lm = lf[a2] - sum(lf[p] for p in parts)
        terms.append((lm, g, m12, tuple(parts)))
    tot = 0.0 + 0.0j
    for lm, g, m12, _ in terms:
        tot += math.exp(lm) * u ** m12 * g ** a3
    return tot, terms


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    th = K4 / 0.55
    print('КОНТРОЛЬ: Z по классам против прямого перечисления M12 (a1=3)')
    for (a2, a3) in ((4, 2), (5, 2), (4, 3)):
        zc, _ = z_classes(3, a2, a3, th)
        zd = z_direct(3, a2, a3, th)
        print(f'  (3,{a2},{a3}): |Z| по классам {abs(zc):.8f}, '
              f'прямо {abs(zd):.8f}, отн. '
              f'{abs(zc-zd)/max(abs(zd),1e-12):.2e}', flush=True)
    print()
    print('КОГЕРЕНТНАЯ ОЦЕНКА против точного |Z| при a1 = 3, a_2 = 9')
    a2 = 9
    _, terms = z_classes(3, a2, 1, th)
    lm_b, g_b, m12_b, parts_b = max(terms, key=lambda t: abs(t[1]))
    print(f'  доминирующий класс: размеры {parts_b}, |G| = {abs(g_b):.4f}, '
          f'ln(число M12) = {lm_b:.4f}')
    print(f"{'a_3':>5}{'ln|Z| точно':>15}{'оценка':>12}{'разница':>11}")
    for a3 in (2, 5, 10, 20, 40):
        zc, _ = z_classes(3, a2, a3, th)
        est = lm_b + a3 * math.log(abs(g_b))
        print(f'{a3:>5}{math.log(abs(zc)):>15.4f}{est:>12.4f}'
              f'{math.log(abs(zc))-est:>11.4f}', flush=True)
    print()
    print('=' * 70)
    print('ТОНКИЙ СКАН: есть ли hbar в полосе, где max|g_d(m)| <= 1')
    best = (1e9, None)
    nposs = 0
    for hb in np.arange(0.390, 0.7805, 0.001):
        t = K4 / hb
        bc, barg, bg = -1e9, None, None
        for m in range(1, 61):
            for d in range(1, 61):
                v = abs(g_md(m, d, t))
                c = math.log(max(v, 1e-300)) / (4 * (m + d))
                if c > bc:
                    bc, barg, bg = c, (m, d), v
        if bc <= 0:
            nposs += 1
        if bc < best[0]:
            best = (bc, hb, barg, bg)
    print(f'  точек скана: {len(np.arange(0.390, 0.7805, 0.001))}')
    print(f'  точек, где сектор гибнет (c <= 0): {nposs}')
    print(f'  минимум c по полосе: {best[0]:.6f} при hbar = {best[1]:.3f}, '
          f'(m,d) = {best[2]}, |g| = {best[3]:.4f}')
    print()
    c = best[0]
    print('СЛЕДСТВИЕ: переход против пространства-времени')
    print(f'  ln|Z| слоистого ~ {c:.4f} N²')
    for lab, k in (('пол многообразных 2 N lnN', 2.0),
                   ('счёт многообразных (Милнор-Том) 4 N lnN', 4.0)):
        # c N^2 = k N ln N  ->  N / ln N = k/c
        r = k / c
        N = 10.0
        for _ in range(200):
            N = r * math.log(N)
        print(f'  против «{lab}»: переход при N ~ {N:.0f}')
    print('  (прежняя оценка перехода была N ~ 1700)')


if __name__ == '__main__':
    main()
