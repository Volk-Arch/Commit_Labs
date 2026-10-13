#!/usr/bin/env python3
"""РОСТ С ПРИСОЕДИНЕНИЕМ К АНТИЦЕПИ: удар по измеренному препятствию.

ПРЕПЯТСТВИЕ (`growth-doc.md`). Звенья нового элемента — ровно
МАКСИМАЛЬНЫЕ элементы набора предков `S`, поэтому `N₀/N ≤ ⟨m⟩`
тождественно. А четырёхмерному посеву нужно `N^(1−2/d)·ln N`: 20.3
звена при `N = 1200` и 47.4 при 4800. Любое испытанное правило даёт
≤ 6.6, потому что при РАВНОМЕРНОМ случайном `S` почти все его элементы
оказываются под другими, и `|max(S)| ≪ |S|`.

МЫСЛЬ. Неравенство `звенья ≤ |S|` обращается в РАВЕНСТВО ровно тогда,
когда `S` — АНТИЦЕПЬ. Тогда звеньев ровно столько, сколько предков, и
ножницы разжимаются по построению.

ПРАВИЛО. На шаге `j` набирается антицепь размера `m` среди уже
существующих элементов (жадно: случайный элемент, затем только
несравнимые со всеми взятыми), и она служит набором предков; прошлое —
замыкание вниз.

ОГОВОРКА. Это НЕ классический последовательный рост: вероятность
зависит не только от `|S|`, значит общая ковариация Райдаута–Соркина
нарушена, и на вывод про КПР отсюда ничего не следует. Проверяется
другой вопрос: может ли правило, нацеленное прямо на препятствие,
удержать четырёхмерие.

ЧТО МЕРИТСЯ: звенья на элемент (обязаны выйти ≈ `m`, это проверка
самого построения), размерность Мирхейма–Мейера на интервалах, и
`C₃`, `C₄` как проверки без свободных чисел.

ЭТАЛОН: сприклинг 4D тем же кодом.

  python3 anti.py
"""
import math
import time

import numpy as np

from grow import mm_dim, c_k, gen_sprinkle, intervals
from bd import layers


def gen_anti(n, m, rs, tries=6):
    A = np.zeros((n, n), dtype=bool)
    for j in range(1, n):
        k = min(int(m), j)
        best = None
        for _ in range(tries):
            avail = np.ones(j, dtype=bool)
            S = []
            while len(S) < k and avail.any():
                c = np.flatnonzero(avail)
                x = int(c[rs.integers(len(c))])
                S.append(x)
                avail &= ~A[:j, x] & ~A[x, :j]
                avail[x] = False
            if best is None or len(S) > len(best):
                best = S
            if len(best) >= k:
                break
        S = np.array(best, dtype=np.int64)
        col = np.zeros(n, dtype=bool)
        col[S] = True
        col[:j] |= A[:j, S].any(axis=1)
        A[:, j] = col
    return A


def stat(gen, n, seeds, root, lo=60, hi=400, want=30, **kw):
    rows = []
    for s in range(seeds):
        A = gen(n, rs=np.random.default_rng(root + 17 * s), **kw)
        nn, L = layers(A, 1)
        c1 = float(np.count_nonzero(A))
        r = c1 / (nn * (nn - 1) / 2.0)
        iv = intervals(A, want, lo, hi, np.random.default_rng(root + 5 + s))
        del A
        if not iv:
            continue
        R = np.array(iv, float)
        m1 = R[:, 0]
        x2 = float((R[:, 1] / (m1 * (m1 - 1))).mean())
        d = mm_dim(x2)
        if not np.isfinite(d) or d <= 0.3:
            continue
        p3 = c_k(3, d) * (m1 * (m1 - 1) * (m1 - 2))
        p4 = c_k(4, d) * (m1 * (m1 - 1) * (m1 - 2) * (m1 - 3))
        rows.append((L[0] / nn, r, mm_dim(r / 2.0) if 0 < r < 1 else np.nan,
                     d, float((R[:, 2] / p3).mean()),
                     float((R[:, 3] / p4).mean())))
    if len(rows) < 2:
        return None
    P = np.array(rows, float)
    return P.mean(axis=0), P.std(axis=0, ddof=1) / math.sqrt(len(P)), len(P)


HDR = (f"{'правило':>26}{'N':>7}{'звен/эл':>10}{'r':>8}{'d всего':>9}"
       f"{'d интерв':>13}{'C₃/пред':>14}{'C₄/пред':>14}")


def show(tag, n, r):
    if r is None:
        print(f'{tag:>26}{n:>7}   интервалов мало')
        return
    m, e, k = r
    print(f'{tag:>26}{n:>7}{m[0]:>10.2f}{m[1]:>8.3f}{m[2]:>9.2f}'
          f'{m[3]:>8.2f}±{e[3]:<4.2f}{m[4]:>8.3f}±{e[4]:<5.3f}'
          f'{m[5]:>8.3f}±{e[5]:<5.3f}')


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print('=' * 110)
    print('ЭТАЛОН')
    print(HDR)
    for n in (800, 1600):
        show('сприклинг 4D', n, stat(gen_sprinkle, n, 6, 300 + n, d=4))
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 110)
    print('ПРАВИЛО АНТИЦЕПИ. Проверка построения: звен/эл обязано быть ≈ m')
    print(HDR)
    for m in (4, 10, 20, 40):
        show(f'антицепь m={m}', 1200, stat(gen_anti, 1200, 5, 600 + m, m=m))
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('ЗАВИСИМОСТЬ ОТ РАЗМЕРА при m = 20 (нужно ~20 звеньев при N=1200)')
    print(HDR)
    for n in (600, 1200, 2400):
        show('антицепь m=20', n, stat(gen_anti, n, 4, 900 + n, m=20))
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
