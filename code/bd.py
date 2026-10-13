#!/usr/bin/env python3
"""ДЕЙСТВИЕ БЕНИНКАСЫ–ДОУКЕРА НА МНОЖЕСТВАХ ИЗ РОСТА.

Второй прибор, независимый от обилий цепей, и он чисто ПОРЯДКОВЫЙ:
длины `ℓ` в нём нет вовсе, это безразмерное число самого частичного
порядка.

    S^(2)/ℏ = 2 ( N − 2N₀ + 4N₁ − 2N₂ )
    S^(4)/ℏ = (4/√6) ( N − N₀ + 9N₁ − 16N₂ + 8N₃ )

где `N_m` — число пар `x ≺ y`, между которыми лежит ровно `m` элементов.
Считается одним целочисленным умножением матриц: если `A` — строгое
отношение, то `(A·A)_xy` и есть число промежуточных элементов, поэтому

    N_m = #{ (x,y) : A_xy = 1 и (A²)_xy = m }.

САМОПРОВЕРКА, встроенная в задачу. В плоском пространстве `R = 0`,
поэтому коэффициенты обязаны почти всё сократить: каждое слагаемое
порядка `N²`, а сумма должна выйти много меньше. Если коэффициенты
перепутаны, сокращения не будет вовсе и ответ останется порядка `N²`.
Это ловит ошибку в знаках без обращения к литературе.

ЧТО СРАВНИВАЕТСЯ. Действие сприклинга при данном `N` — это «эталон
многообразности»; действие множества из роста при том же `N` и той же
измеренной размерности — испытуемое. Литература сообщает, что действие
БД на порождённых ростом множествах не считал никто.

ИЗВЕСТНАЯ ОГОВОРКА. Действие БД не самоусредняется: его флуктуации
растут с `N` (Бенинкаса, Доукер, Глейзер), поэтому разброс по
реализациям здесь такая же часть ответа, как среднее.

  python3 bd.py
"""
import math
import time

import numpy as np

from grow import c_k, mm_dim, gen_sprinkle, gen_percolation, gen_kr, closure
from csg import gen_csg

C2 = (1.0, -2.0, 4.0, -2.0)          # N, N0, N1, N2
C4 = (1.0, -1.0, 9.0, -16.0, 8.0)    # N, N0, N1, N2, N3


def layers(A, top=4):
    """N и N_0..N_{top-1} для строгого отношения A."""
    n = A.shape[0]
    F = A.astype(np.float32)
    B = np.rint(F @ F).astype(np.int64)      # число промежуточных
    rel = A
    out = []
    for m in range(top):
        out.append(int(np.count_nonzero(rel & (B == m))))
    return n, out


def action(A, d):
    n, L = layers(A, 4)
    if d == 2:
        s = C2[0] * n + sum(C2[i + 1] * L[i] for i in range(3))
        return 2.0 * s, n, L
    s = C4[0] * n + sum(C4[i + 1] * L[i] for i in range(4))
    return (4.0 / math.sqrt(6.0)) * s, n, L


def run(gen, n, dim, seeds, base, **kw):
    S, D, RAW = [], [], []
    for s in range(seeds):
        rs = np.random.default_rng(base + 7919 * s)
        A = gen(n, rs=rs, **kw)
        a, nn, L = action(A, dim)
        S.append(a)
        RAW.append(L)
        # размерность Мирхейма–Мейера всего множества
        c1 = float(np.count_nonzero(A))
        r = c1 / (nn * (nn - 1) / 2.0)
        D.append(mm_dim(r / 2.0) if 0 < r < 1 else np.nan)
    S = np.array(S, float)
    D = np.array(D, float)
    return S, D, np.array(RAW, float)


def show(tag, S, D, n):
    m, sd = S.mean(), S.std(ddof=1) if len(S) > 1 else 0.0
    se = sd / math.sqrt(len(S))
    dm = np.nanmean(D)
    print(f'{tag:>34}{n:>7}{dm:>8.2f}{m:>14.1f}{se:>9.1f}{sd:>11.1f}'
          f'{m/n:>11.3f}{m/n**2:>12.5f}')


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print('=' * 108)
    print('САМОПРОВЕРКА: сокращаются ли коэффициенты в плоском пространстве?')
    print('(если да, S/N² → 0 с ростом N; если коэффициенты неверны, '
          'останется O(1))')
    print(f"{'что':>34}{'N':>7}{'d':>8}{'S/ℏ':>14}{'±':>9}{'разброс':>11}"
          f"{'S/N':>11}{'S/N²':>12}")
    for dd in (2, 4):
        for n in (400, 800, 1600):
            S, D, _ = run(gen_sprinkle, n, dd, 12, 100 + n + dd, d=dd)
            show(f'сприклинг {dd}D', S, D, n)
        print()
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print('=' * 108)
    print('НЕМНОГООБРАЗНЫЕ: порядок Клейтмана–Ротшильда (3 слоя)')
    print(f"{'что':>34}{'N':>7}{'d':>8}{'S/ℏ':>14}{'±':>9}{'разброс':>11}"
          f"{'S/N':>11}{'S/N²':>12}")
    for n in (400, 800, 1600):
        for dd in (2, 4):
            S, D, _ = run(gen_kr, n, dd, 6, 500 + n + dd)
            show(f'КР 3 слоя, оценка {dd}D', S, D, n)
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 108)
    print('РОСТ: степенное семейство, настроенное на d = 2 (из scale.py)')
    print(f"{'что':>34}{'N':>7}{'d':>8}{'S/ℏ':>14}{'±':>9}{'разброс':>11}"
          f"{'S/N':>11}{'S/N²':>12}")
    for n, a in ((400, 0.44), (800, 0.4486), (1600, 0.4699)):
        for dd in (2, 4):
            S, D, _ = run(gen_csg, n, dd, 8, 900 + n + dd,
                          draw_m=(lambda j, rs, a=a: max(1, int(round(j ** a)))))
            show(f'рост a={a}, оценка {dd}D', S, D, n)
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('РОСТ: перколяция, настроенная по размерности')
    print(f"{'что':>34}{'N':>7}{'d':>8}{'S/ℏ':>14}{'±':>9}{'разброс':>11}"
          f"{'S/N':>11}{'S/N²':>12}")
    for n in (400, 800, 1600):
        for p in (0.0316, 0.2):
            S, D, _ = run(gen_percolation, n, 2, 6, 1300 + n, p=p)
            show(f'перколяция p={p}, оценка 2D', S, D, n)
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
