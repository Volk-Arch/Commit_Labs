#!/usr/bin/env python3
"""ДВА ПРИБОРА НА ОДНОМ ОБЪЕКТЕ: обилия цепей против действия БД.

РАСХОЖДЕНИЕ, которое надо разрешить. На множествах из роста, настроенных
на `d = 2`:
  * обилия цепей `C_3`, `C_4` ложатся в 2…13 % от предсказания — ПРОХОДЯТ;
  * действие БД даёт `S/N ≈ 8` вместо нуля — ПРОВАЛИВАЕТ.

Но это считалось на РАЗНЫХ объектах: обилия — на интервалах по 120…900
элементов, действие — на всём множестве. Сравнение нечестное.

ЗДЕСЬ ОБА ПРИБОРА НА ОДНИХ И ТЕХ ЖЕ ИНТЕРВАЛАХ, и рядом — интервалы
сприклинга того же размера как эталон. Если тот же интервал проходит
обилия и проваливает действие, то приборы не равносильны и действие
строго сильнее: обилия ловят долю упорядоченных пар (крупная
структура), действие ловит обилия МАЛЫХ интервалов `N_0..N_3`
(микроструктура звеньев).

ВТОРАЯ ЧАСТЬ: существует ли связь, при которой `S/N → 0`? Это и есть
условие «микроструктура как у многообразия». Сканируются все пять
семейств по всему диапазону.

Шумовой пол известен из `bd.py`: у сприклинга `σ(S) ≈ 260` при
`N = 1600`, то есть `|S/N| ≈ 0.16`. Всё, что выше единицы, — сигнал.

  python3 density.py
"""
import math
import time

import numpy as np

from grow import mm_dim, gen_sprinkle, gen_percolation, intervals, spectrum
from csg import gen_csg
from bd import action, C2, C4


def iv_rows(A, want, lo, hi, rs, dim, tries=6000):
    """Для каждого интервала: размер, C_2/(n(n-1)), действие."""
    n = A.shape[0]
    out = []
    for _ in range(tries):
        if len(out) >= want:
            break
        a = int(rs.integers(0, n - 1))
        fut = np.flatnonzero(A[a])
        if len(fut) < lo:
            continue
        b = int(fut[rs.integers(len(fut))])
        idx = np.flatnonzero(A[a] & A[:, b])
        m = len(idx)
        if not (lo <= m <= hi):
            continue
        sub = A[np.ix_(idx, idx)]
        s, _, _ = action(sub, dim)
        c2 = float(np.count_nonzero(sub))
        out.append((m, c2 / (m * (m - 1)), s))
    return out


def arm(gen, n, dim, seeds, root, want=40, lo=60, hi=400, **kw):
    per = []
    for s in range(seeds):
        A = gen(n, rs=np.random.default_rng(root + 17 * s), **kw)
        R = iv_rows(A, want, lo, hi, np.random.default_rng(root + 5 + s), dim)
        del A
        if len(R) < 3:
            continue
        R = np.array(R, float)
        per.append((R[:, 0].mean(), mm_dim(R[:, 1].mean()),
                    (R[:, 2] / R[:, 0]).mean(), R[:, 2].mean()))
    if len(per) < 2:
        return None
    P = np.array(per, float)
    mu = P.mean(axis=0)
    se = P.std(axis=0, ddof=1) / math.sqrt(len(P))
    return mu, se, len(P)


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print('=' * 100)
    print('ЧАСТЬ 1. Оба прибора на ОДНИХ интервалах (60…400 элементов)')
    print(f"{'что':>36}{'⟨n⟩':>7}{'d обилий':>11}{'S/ℏ на интервал':>20}"
          f"{'S/n':>16}")
    rows = [
        ('сприклинг 2D (эталон)', gen_sprinkle, 1600, 2, dict(d=2), 10),
        ('сприклинг 4D (эталон)', gen_sprinkle, 1600, 4, dict(d=4), 10),
        ('рост a=0.4699, оценка 2D', gen_csg, 1600, 2,
         dict(draw_m=lambda j, rs: max(1, int(round(j ** 0.4699)))), 8),
        ('рост a=0.4699, оценка 4D', gen_csg, 1600, 4,
         dict(draw_m=lambda j, rs: max(1, int(round(j ** 0.4699)))), 8),
        ('перколяция p=0.0316, 2D', gen_percolation, 1600, 2,
         dict(p=0.0316), 8),
        ('перколяция p=0.0316, 4D', gen_percolation, 1600, 4,
         dict(p=0.0316), 8),
    ]
    for tag, gen, n, dim, kw, sd in rows:
        r = arm(gen, n, dim, sd, 4000 + 13 * dim + len(tag), **kw)
        if r is None:
            print(f'{tag:>36}   интервалов мало')
            continue
        mu, se, k = r
        print(f'{tag:>36}{mu[0]:>7.0f}{mu[1]:>7.2f}±{se[1]:<3.2f}'
              f'{mu[3]:>13.1f}±{se[3]:<6.1f}{mu[2]:>10.3f}±{se[2]:<5.3f}')
        print(f'    ({time.time()-t0:.0f} с, реализаций {k})', flush=True)
    print()
    print('=' * 100)
    print('ЧАСТЬ 2. Есть ли связь с S/N → 0? Скан по всем семействам, N=1200')
    print(f"{'семейство':>26}{'параметр':>12}{'d всего мн.':>13}"
          f"{'S²/N':>12}{'S⁴/N':>12}")
    N, SD = 1200, 5

    def pw(a):
        return lambda j, rs: max(1, int(round(j ** a)))

    def const(m):
        return lambda j, rs: m

    def pois(mu):
        return lambda j, rs: max(1, int(rs.poisson(mu)))

    def frac(c):
        return lambda j, rs: max(1, int(round(c * j)))

    best = []
    fams = [
        ('перколяция p', [('p', v) for v in
                          (0.003, 0.01, 0.0316, 0.1, 0.3, 0.6)], None),
        ('степенное j^a', [('a', v) for v in
                           (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)], pw),
        ('постоянное m', [('m', v) for v in (1, 2, 4, 10, 30, 90)], const),
        ('Пуассон mu', [('mu', v) for v in (1, 2, 5, 15, 45, 120)], pois),
        ('доля c*j', [('c', v) for v in
                      (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)], frac),
    ]
    for fname, pts, mk in fams:
        for pname, v in pts:
            S2, S4, DD = [], [], []
            for s in range(SD):
                rs = np.random.default_rng(7000 + 31 * s + int(1e4 * v))
                A = (gen_percolation(N, p=v, rs=rs) if mk is None
                     else gen_csg(N, draw_m=mk(v), rs=rs))
                a2, _, _ = action(A, 2)
                a4, _, _ = action(A, 4)
                r = float(np.count_nonzero(A)) / (N * (N - 1) / 2.0)
                S2.append(a2 / N); S4.append(a4 / N)
                DD.append(mm_dim(r / 2.0) if 0 < r < 1 else np.nan)
                del A
            m2, m4 = float(np.mean(S2)), float(np.mean(S4))
            e2 = float(np.std(S2, ddof=1)) / math.sqrt(SD)
            print(f'{fname:>26}{pname}={v:<10.4g}{np.nanmean(DD):>13.2f}'
                  f'{m2:>8.2f}±{e2:<4.2f}{m4:>12.2f}')
            best.append((abs(m2), fname, pname, v, m2, m4,
                         float(np.nanmean(DD))))
        print(f'    ({time.time()-t0:.0f} с)', flush=True)
    print()
    best.sort()
    print('МИНИМУМ |S²/N| по всему скану (ближайшее к многообразности):')
    for b in best[:6]:
        print(f'    {b[1]:>24} {b[2]}={b[3]:<9.4g} d={b[6]:.2f}  '
              f'S²/N={b[4]:+.2f}  S⁴/N={b[5]:+.2f}')
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
