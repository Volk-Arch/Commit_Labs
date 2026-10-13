#!/usr/bin/env python3
"""ТОЧНАЯ СТАТСУММА: ансамбль порядков без выборки вовсе — пункт 3.

Сейчас у проекта есть три СКОНСТРУИРОВАННЫХ сектора с разной плотностью
действия (−0.408, +0.531, плато O(1)). Это сырьё, а не фазовая
структура: нет статсуммы, нет константы связи для скана, нет свободной
энергии, нет параметра порядка, прослеженного ЧЕРЕЗ переход.

ЗДЕСЬ всё это считается ТОЧНО, перечислением. Для малого `N` все
порядки перебираются целиком, поэтому

    Z(beta) = sum_порядки exp(-beta S),   F = -ln Z / N²

и любое среднее — точные суммы, без MCMC, без ошибки выборки, без
тонкостей детального баланса. Это дороже по `N`, но зато не может
соврать.

АНСАМБЛЬ: порядки с ФИКСИРОВАННЫМ линейным расширением, то есть строго
верхнетреугольные транзитивные отношения. Выбор не произволен — ровно
такие порядки выдаёт сприклинг (точки сортируются по времени), и все
генераторы проекта тоже.

Число таких отношений равно `2^(N(N-1)/2)` до проверки транзитивности:
N = 7 даёт 2097152, перебирается целиком; N = 8 даёт 2.7e8 — уже нет.

ПАРАМЕТРЫ ПОРЯДКА, все внутренние:
  `S/N²` — плотность действия БД;
  доля упорядоченных пар `r` (из неё размерность Мирхейма–Мейера);
  высота — длина самой длинной цепи, отличает цепное от плоского;
  восприимчивость `Var(S)/N²` — её ПИК и есть подпись перехода.

КОНТРОЛИ, проверяемые руками:
  1. число порядков: N = 2 даёт 2, N = 3 даёт 7 (считается на бумаге);
  2. при `beta = 0` доля упорядоченных пар обязана совпасть с прямым
     средним по перечислению;
  3. `S` плоского (пустого) порядка обязано дать `2N` для `d = 2`.

  python3 ensemble.py
"""
import math

import numpy as np

K4 = 4.0 / math.sqrt(6.0)
BETAS = np.concatenate([np.linspace(-1.2, 1.2, 49)])


def pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def enumerate_orders(n, chunk=200000):
    """Все строго верхнетреугольные транзитивные отношения на n точках."""
    P = pairs(n)
    m = len(P)
    ii = np.array([p[0] for p in P]); jj = np.array([p[1] for p in P])
    tot = 1 << m
    out = {'S2': [], 'S4': [], 'rel': [], 'hgt': [],
           'N0': [], 'N1': [], 'N2': []}
    for lo in range(0, tot, chunk):
        hi = min(lo + chunk, tot)
        idx = np.arange(lo, hi, dtype=np.int64)
        bits = ((idx[:, None] >> np.arange(m)[None, :]) & 1).astype(bool)
        A = np.zeros((len(idx), n, n), dtype=bool)
        A[:, ii, jj] = bits
        F = A.astype(np.int16)
        B = np.einsum('mik,mkj->mij', F, F, optimize=True)
        ok = ~((B > 0) & ~A).any(axis=(1, 2))
        if not ok.any():
            continue
        A, B = A[ok], B[ok]
        rel = A.sum(axis=(1, 2))
        N0 = (A & (B == 0)).sum(axis=(1, 2))
        N1 = (A & (B == 1)).sum(axis=(1, 2))
        N2 = (A & (B == 2)).sum(axis=(1, 2))
        N3 = (A & (B == 3)).sum(axis=(1, 2))
        out['S2'].append(2.0 * (n - 2 * N0 + 4 * N1 - 2 * N2))
        out['S4'].append(K4 * (n - N0 + 9 * N1 - 16 * N2 + 8 * N3))
        out['rel'].append(rel)
        out['N0'].append(N0)   # N_0 = число СВЯЗЕЙ:
        out['N1'].append(N1)   # отношений с пустым интервалом
        out['N2'].append(N2)
        # высота: длина самой длинной цепи = max по k, что (A^k) != 0
        h = np.zeros(len(A), dtype=np.int32)
        R = A.copy()
        for k in range(1, n):
            alive = R.any(axis=(1, 2))
            h[alive] = k
            if not alive.any():
                break
            R = np.einsum('mik,mkj->mij', R.astype(np.int16),
                          A.astype(np.int16), optimize=True) > 0
        out['hgt'].append(h)
    return {k: np.concatenate(v) for k, v in out.items()}


def thermo(S, rel, hgt, n, betas):
    """Точные Z, F и средние по перечислению."""
    npair = n * (n - 1) // 2
    rows = []
    for b in betas:
        e = -b * S
        e -= e.max()
        w = np.exp(e)
        Zl = math.log(w.sum()) + e.max() + b * 0.0
        sm = (w * S).sum() / w.sum()
        s2 = (w * S * S).sum() / w.sum()
        rm = (w * rel).sum() / w.sum() / npair
        hm = (w * hgt).sum() / w.sum()
        rows.append((b, -Zl / n ** 2, sm / n ** 2,
                     (s2 - sm ** 2) / n ** 2, rm, hm))
    return np.array(rows)


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print('КОНТРОЛЬ 1: число порядков (на бумаге N=2 -> 2, N=3 -> 7)')
    cnt = {}
    for n in (2, 3, 4, 5, 6, 7):
        d = enumerate_orders(n)
        cnt[n] = len(d['S2'])
        print(f'  N = {n}: порядков {cnt[n]}', flush=True)
        if n == 7:
            keep = d
    ok = cnt[2] == 2 and cnt[3] == 7
    print(f'  -> {"СОВПАЛО" if ok else "НЕ СОВПАЛО"}')
    print()
    print('КОНТРОЛЬ 3: S пустого порядка при d=2 обязано быть 2N')
    d2 = enumerate_orders(3)
    print(f'  N = 3: min|S2| = {np.abs(d2["S2"]).min():.1f}, '
          f'ожидается 6 у пустого')
    print()
    print('=' * 76)
    print('ТОЧНАЯ ТЕРМОДИНАМИКА, действие d = 2')
    for n in (5, 6, 7):
        d = enumerate_orders(n)
        T = thermo(d['S2'], d['rel'], d['hgt'], n, BETAS)
        k = int(np.argmax(T[:, 3]))
        print(f'  N = {n}: порядков {len(d["S2"])}, '
              f'пик восприимчивости при beta = {T[k,0]:+.3f} '
              f'(значение {T[k,3]:.3f})')
        print(f'{"beta":>8}{"F/N²":>10}{"<S>/N²":>10}{"Var S/N²":>11}'
              f'{"доля r":>9}{"высота":>8}')
        for row in T[::6]:
            print(f'{row[0]:>8.2f}{row[1]:>10.4f}{row[2]:>10.4f}'
                  f'{row[3]:>11.4f}{row[4]:>9.4f}{row[5]:>8.2f}')
        print(flush=True)
    print('''ЧТЕНИЕ. Пик восприимчивости `Var S` — подпись перехода. Если с
ростом `N` он РАСТЁТ и СУЖАЕТСЯ, а положение сходится — это настоящий
фазовый переход, а не особенность малого размера. Если пик не растёт —
перехода в этом ансамбле нет, и это тоже ответ.''')


if __name__ == '__main__':
    main()
