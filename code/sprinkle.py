#!/usr/bin/env python3
"""СПРИКЛИНГ В ПРОИЗВОЛЬНОЙ РАЗМЕРНОСТИ — починка эталона.

ОШИБКА, которую это исправляет. В `layers.py` я вызвал `grow.gen_sprinkle`
с `d = 3` и `d = 5`, а он поддерживает только 2 и 4: остальное молча
падало в четырёхмерную ветку, и три строки эталона вышли одинаковыми
(0.352, 0.212, 0.155) с размерностью 4.0 во всех. Применение генератора
вне области определения, и опознаётся оно по тому, что ответ НЕ ЗАВИСИТ
от заявленного параметра.

ЧТО ЗДЕСЬ. Ромб между (0,0⃗) и (1,0⃗) в `d` измерениях: `t ∈ [0,1]`,
`|x⃗| ≤ R(t) = min(t, 1−t)`. Плотность по объёму даёт `p(t) ∝ R(t)^(d−1)`,
а внутри — равномерно по шару радиуса `R(t)` в `d−1` измерениях.
Отношение: `(t_j − t_i) > |x⃗_j − x⃗_i|`.

Расстояния — через тождество Грама по блокам строк
(`|x_i − x_j|² = s_i + s_j − 2(XXᵀ)_ij`), чтобы не держать массив
`n×n×(d−1)`.

ПРОВЕРКА: при `d = 2` и `d = 4` обязано совпасть с прежним генератором
(`r` = 0.501 и 0.101, `N₁/N₀` = 0.825 и 0.352) — иначе новый неверен.

ЗАЧЕМ. Эталон перестаёт быть двумя точками и становится КРИВОЙ: у
сприклинга и доля упорядоченных пар `r`, и отношения `N_m/N₀` суть
функции ОДНОЙ размерности, значит в плоскости `(r, N₁/N₀)` сприклинги
лежат на однопараметрической кривой. Вопрос о росте тогда ставится
точно: попадает ли хоть одно правило на эту кривую в точке `d = 4`?

  python3 sprinkle.py
"""
import math
import time

import numpy as np

from grow import mm_dim, gen_sprinkle as old_sprinkle
from bd import layers, action


def sprinkle_d(n, d, rs, block=512):
    """Сприклинг в причинный ромб размерности d ≥ 2."""
    s = d - 1                                  # пространственных осей
    u = rs.random(n)
    v = rs.random(n)
    half = 0.5 * v ** (1.0 / d)
    t = np.where(u < 0.5, half, 1.0 - half)
    R = np.minimum(t, 1.0 - t)
    if s == 1:
        X = (rs.random((n, 1)) * 2.0 - 1.0) * R[:, None]
    else:
        g = rs.normal(size=(n, s))
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        rad = R * rs.random(n) ** (1.0 / s)
        X = g * rad[:, None]
    o = np.argsort(t)
    t, X = t[o], X[o]
    sq = (X * X).sum(axis=1)
    A = np.zeros((n, n), dtype=bool)
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        d2 = sq[i0:i1, None] + sq[None, :] - 2.0 * (X[i0:i1] @ X.T)
        np.maximum(d2, 0.0, out=d2)
        dt = t[None, :] - t[i0:i1, None]
        A[i0:i1] = (dt > 0) & (dt * dt > d2)
    return A


def line(tag, A):
    n, L = layers(A, 4)
    c1 = float(np.count_nonzero(A))
    r = c1 / (n * (n - 1) / 2.0)
    dd = mm_dim(r / 2.0) if 0 < r < 1 else float('nan')
    a2, _, _ = action(A, 2)
    a4, _, _ = action(A, 4)
    return (dd, r, L[0] / n, L[1] / max(L[0], 1), L[2] / max(L[0], 1),
            L[3] / max(L[0], 1), a2 / n, a4 / n)


def avg(gen, n, seeds, root, **kw):
    P = []
    for s in range(seeds):
        A = gen(n, rs=np.random.default_rng(root + 17 * s), **kw)
        P.append(line('', A))
        del A
    P = np.array(P, float)
    return (P.mean(axis=0),
            P.std(axis=0, ddof=1) / math.sqrt(len(P)) if len(P) > 1
            else np.zeros(P.shape[1]))


HDR = (f"{'что':>26}{'d Мирх.-М.':>12}{'r':>9}{'N₀/N':>9}"
       f"{'N₁/N₀':>14}{'N₂/N₀':>14}{'N₃/N₀':>14}{'S²/N':>9}{'S⁴/N':>9}")


def show(tag, m, e):
    print(f'{tag:>26}{m[0]:>12.3f}{m[1]:>9.4f}{m[2]:>9.2f}'
          f'{m[3]:>8.3f}±{e[3]:<5.3f}{m[4]:>8.3f}±{e[4]:<5.3f}'
          f'{m[5]:>8.3f}±{e[5]:<5.3f}{m[6]:>9.2f}{m[7]:>9.2f}')


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    N = 1200
    print('=' * 116)
    print('ПРОВЕРКА нового генератора против старого (обязаны совпасть)')
    print(HDR)
    for dd in (2, 4):
        m, e = avg(old_sprinkle, N, 8, 400 + dd, d=dd)
        show(f'прежний {dd}D', m, e)
        m, e = avg(sprinkle_d, N, 8, 400 + dd, d=dd)
        show(f'новый {dd}D', m, e)
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 116)
    print('ЭТАЛОННАЯ КРИВАЯ: сприклинг по размерности, N = 1200')
    print(HDR)
    ref = {}
    for dd in (2, 3, 4, 5, 6, 8):
        m, e = avg(sprinkle_d, N, 8, 700 + dd, d=dd)
        show(f'сприклинг {dd}D', m, e)
        ref[dd] = m
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('ПРОВЕРКА арифметики: r должно равняться 2·c₂(d)')
    from grow import c_k
    for dd in (2, 3, 4, 5, 6, 8):
        print(f'    d={dd}: r измерено {ref[dd][1]:.4f}, '
              f'2c₂(d) = {2*c_k(2, dd):.4f}, '
              f'd из r = {ref[dd][0]:.3f}')
    print()
    print('ЦЕЛЬ для роста: четырёхмерная точка эталона')
    m = ref[4]
    print(f'    r = {m[1]:.4f},  звеньев на элемент = {m[2]:.2f},  '
          f'N₁/N₀ = {m[3]:.3f} (УБЫВАЕТ),  |S⁴/N| = {abs(m[7]):.2f}')
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
