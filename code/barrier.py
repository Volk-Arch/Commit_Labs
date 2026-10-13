#!/usr/bin/env python3
"""ЕСТЬ ЛИ БАРЬЕР? Локальный наклон действия вокруг сприклинга.

ВОПРОС. В `suppress.py` показано, что у слоистых порядков действие много
ниже (`S/N² → −1/2` в двух измерениях, `−1/(2√6)` в четырёх), то есть
путь вниз СУЩЕСТВУЕТ. Но существует ли он ЛОКАЛЬНО? Если у сприклинга
всякий элементарный ход увеличивает действие, многообразие —
устойчивая точка, и тогда понятно, как непрерывность выживает против
энтропии. Если всякий ход уменьшает — барьера нет.

Это первый раз, когда вопрос вообще можно поставить: нужен прибор,
считающий действие на ЛЮБОМ порядке, а не только на сприклинге.

ЭЛЕМЕНТАРНЫЕ ХОДЫ, и оба сохраняют частичный порядок.
  * УБРАТЬ ЗВЕНО `x ≺ y` (пара с пустым интервалом). Транзитивность не
    нарушается: если бы было `x ≺ k ≺ y`, интервал не был бы пуст.
  * ДОБАВИТЬ ОТНОШЕНИЕ между НЕСРАВНИМЫМИ `x, y` и замкнуть:
    `(прошлое(x) ∪ {x}) × (будущее(y) ∪ {y})`. Цикла не возникает,
    поскольку `y ⊀ x` по условию несравнимости.

ЧТО СЧИТАЕТСЯ. Для каждого хода точное `ΔS` пересчётом слоёв. Знак
важнее величины: действие само отрицательно у слоистых, поэтому
«вниз» — это к слоистости.

ЭТАЛОНЫ. Те же ходы на слоистом порядке: там спуск к слоистости обязан
быть виден, иначе прибор ничего не мерит.

  python3 barrier.py
"""
import math
import time

import numpy as np

from grow import mm_dim
from bd import action, layers
from sprinkle import sprinkle_d
from suppress import gen_layered


def links_of(A):
    F = A.astype(np.float32)
    B = np.rint(F @ F).astype(np.int64)
    return np.argwhere(A & (B == 0))


def incomparable(A, rs, k):
    n = A.shape[0]
    out = []
    for _ in range(40 * k):
        if len(out) >= k:
            break
        i, j = int(rs.integers(n)), int(rs.integers(n))
        if i == j or A[i, j] or A[j, i]:
            continue
        out.append((i, j))
    return out


def dS_remove(A, x, y, dim):
    s0, _, _ = action(A, dim)
    A[x, y] = False
    s1, _, _ = action(A, dim)
    A[x, y] = True
    return s1 - s0


def dS_add(A, x, y, dim):
    s0, _, _ = action(A, dim)
    p = A[:, x].copy(); p[x] = True
    f = A[y, :].copy(); f[y] = True
    old = A[np.ix_(p, f)].copy()
    A[np.ix_(p, f)] = True
    s1, _, _ = action(A, dim)
    A[np.ix_(p, f)] = old
    return s1 - s0, int(p.sum() * f.sum() - old.sum())


def survey(tag, A, dim, rs, k=140):
    lk = links_of(A)
    if len(lk) == 0:
        print(f'{tag:>30}  звеньев нет')
        return
    sel = lk[rs.choice(len(lk), size=min(k, len(lk)), replace=False)]
    R = np.array([dS_remove(A, int(x), int(y), dim) for x, y in sel], float)
    inc = incomparable(A, rs, k)
    AD, NA = [], []
    for x, y in inc:
        d, cnt = dS_add(A, x, y, dim)
        AD.append(d); NA.append(cnt)
    AD = np.array(AD, float) if AD else np.array([np.nan])
    NA = np.array(NA, float) if NA else np.array([np.nan])
    s0, _, _ = action(A, dim)
    print(f'{tag:>30}{s0:>12.1f}'
          f'{R.mean():>11.2f}±{R.std(ddof=1)/math.sqrt(len(R)):<6.2f}'
          f'{100*(R < 0).mean():>8.0f}%'
          f'{np.nanmean(AD):>12.2f}±{np.nanstd(AD, ddof=1)/math.sqrt(max(len(AD),1)):<7.2f}'
          f'{100*np.nanmean(AD < 0):>8.0f}%{np.nanmean(NA):>10.1f}')


HDR = (f"{'что':>30}{'S/ℏ':>12}{'ΔS убрать звено':>18}{'<0':>9}"
       f"{'ΔS добавить отн.':>20}{'<0':>9}{'отн. за ход':>11}")


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    N = 400
    for dim in (2, 4):
        print('=' * 118)
        print(f'РАЗМЕРНОСТЬ ДЕЙСТВИЯ d = {dim}, N = {N}')
        print(HDR)
        for sd in (2, 4):
            A = sprinkle_d(N, sd, np.random.default_rng(900 + sd))
            survey(f'сприклинг {sd}D', A, dim,
                   np.random.default_rng(11 + sd))
            del A
        for parts, q in (((1, 2, 1), 0.5), ((1, 2, 1), 0.15), ((1, 1), 0.5)):
            A, _ = gen_layered(N, parts, q,
                               np.random.default_rng(300 + int(100 * q)))
            survey(f'слоистый {parts} q={q}', A, dim,
                   np.random.default_rng(77 + int(100 * q)))
            del A
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
        print()
    print('=' * 118)
    print('ЗАВИСИМОСТЬ ОТ РАЗМЕРА: растёт ли барьер? (сприклинг 4D, d=4)')
    print(f"{'N':>8}{'S/ℏ':>12}{'ΔS убрать':>14}{'ΔS добавить':>16}"
          f"{'отн. за ход':>14}{'ΔS добав./отн.':>17}")
    for n in (200, 400, 800):
        A = sprinkle_d(n, 4, np.random.default_rng(1700 + n))
        rs = np.random.default_rng(55 + n)
        lk = links_of(A)
        sel = lk[rs.choice(len(lk), size=min(120, len(lk)), replace=False)]
        R = np.array([dS_remove(A, int(x), int(y), 4) for x, y in sel], float)
        AD, NA = [], []
        for x, y in incomparable(A, rs, 120):
            d, c = dS_add(A, x, y, 4)
            AD.append(d); NA.append(c)
        s0, _, _ = action(A, 4)
        del A
        AD, NA = np.array(AD, float), np.array(NA, float)
        print(f'{n:>8}{s0:>12.1f}{R.mean():>14.2f}{AD.mean():>16.2f}'
              f'{NA.mean():>14.1f}{AD.mean()/max(NA.mean(),1e-9):>17.3f}')
        print(f'    ({time.time()-t0:.0f} с)', flush=True)
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
