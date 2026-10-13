#!/usr/bin/env python3
"""Сито по собственному времени: обрезка, которая ничего не ломает.

locality-doc.md: локальности нет, и все прежние способы её получить
ломали либо транзитивность (кадр по времени), либо лоренц-инвариантность
(обрезка по пространственному расстоянию).

Но просеивать можно по СОБСТВЕННОМУ ВРЕМЕНИ, а оно инвариантно:

    tau = sqrt(dt^2 - dx^2)

Правило сита: события не связаны, если tau < s.

Два свойства достаются бесплатно:

  лоренц-инвариантность — tau инвариантно по определению;
  транзитивность — для времениподобных интервалов верно ОБРАТНОЕ
    неравенство треугольника, tau(A,C) >= tau(A,B) + tau(B,C).
    Значит если две связи сито прошли, составная пройдёт тем более.

Для сравнения — обратное сито (tau > s, то есть «не помним далёкое») и
обрезка по пространственному расстоянию, которые эти свойства ломают.

  python3 sieve.py
  python3 sieve.py --n 2500
"""
import argparse
import math

import numpy as np

from core import (dimension, frac_pairs, spectral, height, links_of,
                  transitivity_violations)
from coneworld import sprinkle


def rel_sieve(X, s=None, mode='min'):
    """Причинный порядок с ситом. R[i,j] = True, если j раньше i."""
    dt = X[:, None, 0] - X[None, :, 0]
    dx = np.linalg.norm(X[:, None, 1:] - X[None, :, 1:], axis=-1)
    R = dt > dx
    if s is not None:
        tau2 = np.where(R, dt * dt - dx * dx, 0.0)
        if mode == 'min':
            R &= tau2 >= s * s          # не связаны, если ближе чем s
        elif mode == 'max':
            R &= tau2 <= s * s          # не связаны, если дальше чем s
        elif mode == 'space':
            R &= dx <= s                # обрезка по расстоянию (для сравнения)
    np.fill_diagonal(R, False)
    return np.asarray(R, bool)


def interior(X, R, frac=0.3):
    """Доля событий области, у которых ВСЕ прямые связи внутри."""
    L = links_of(R)
    r = np.abs(X[:, 0]) + np.linalg.norm(X[:, 1:], axis=1)
    m = r <= np.quantile(r, frac)
    idx = np.flatnonzero(m)
    if len(idx) < 10:
        return float('nan')
    out = (L[np.ix_(idx, ~m)].any(1)) | (L[np.ix_(~m, idx)].any(0))
    return float((~out).mean())


def boost(X, beta):
    g = 1 / math.sqrt(1 - beta * beta)
    Y = X.copy()
    Y[:, 0] = g * (X[:, 0] - beta * X[:, 1])
    Y[:, 1] = g * (X[:, 1] - beta * X[:, 0])
    return Y


def main():
    ap = argparse.ArgumentParser(description='Сито по собственному времени')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()

    print(f'N={a.n}, засев в причинный ромб, семян {a.seeds}')
    print('ромб имеет собственное время 2, поэтому сито s в тех же единицах')
    print()
    print(f"{'сито':>22}{'доля пар':>10}{'разм.':>8}{'выс.':>7}{'спектр':>9}"
          f"{'наруш.транз.':>14}{'связей/соб.':>12}{'внутр.':>8}")

    rows = {}
    cases = [('без сита', None, 'min')]
    cases += [(f'tau >= {s}', s, 'min') for s in (0.1, 0.3, 0.6, 1.0)]
    cases += [(f'tau <= {s}', s, 'max') for s in (1.0, 0.6)]
    cases += [(f'расстояние <= {s}', s, 'space') for s in (0.6, 0.3)]
    for lbl, s, mode in cases:
        acc = []
        for t in range(a.seeds):
            rs = np.random.default_rng(700 + 17 * t)
            X = sprinkle(a.n, 4, rs)
            R = rel_sieve(X, s, mode)
            f = frac_pairs(R)
            try:
                d = dimension(R=R) if 0 < f < 1 else float('nan')
            except Exception:
                d = float('nan')
            acc.append((f, d, height(R)[0], spectral(R),
                        transitivity_violations(R), links_of(R).sum() / a.n,
                        interior(X, R)))
        m = np.array(acc, float).mean(0)
        rows[lbl] = m
        print(f'{lbl:>22}{m[0]:>10.4f}{m[1]:>8.3f}{m[2]:>7.1f}{m[3]:>9.4f}'
              f'{m[4]:>14.0f}{m[5]:>12.1f}{m[6]:>8.4f}')

    print()
    print('ЛОРЕНЦ-ИНВАРИАНТНОСТЬ: доля отношений, меняющихся при бусте')
    print(f"{'сито':>22}{'буст 0.3':>12}{'буст 0.6':>12}{'буст 0.9':>12}")
    rs = np.random.default_rng(700)
    X = sprinkle(a.n, 4, rs)
    for lbl, s, mode in [('без сита', None, 'min'), ('tau >= 0.3', 0.3, 'min'),
                         ('tau <= 1.0', 1.0, 'max'),
                         ('расстояние <= 0.6', 0.6, 'space')]:
        R0 = rel_sieve(X, s, mode)
        cells = []
        for b in (0.3, 0.6, 0.9):
            R1 = rel_sieve(boost(X, b), s, mode)
            cells.append((R0 != R1).sum() / R0.size)
        print(f'{lbl:>22}' + ''.join(f'{c:>12.6f}' for c in cells))

    print()
    print('МАСШТАБИРОВАНИЕ СВЯЗЕЙ: растут ли они со размером мира')
    print(f"{'сито':>22}{'показатель по N':>18}{'без сита было':>16}")
    for lbl, s, mode in [('без сита', None, 'min'), ('tau >= 0.3', 0.3, 'min'),
                         ('tau >= 0.6', 0.6, 'min')]:
        ns, ls = [], []
        for N in (256, 512, 1024, 2048):
            v = []
            for t in range(2):
                rs = np.random.default_rng(800 + 17 * t)
                R = rel_sieve(sprinkle(N, 4, rs), s, mode)
                v.append(links_of(R).sum() / N)
            ns.append(N)
            ls.append(float(np.mean(v)))
        p = np.polyfit(np.log(ns), np.log(np.maximum(ls, 1e-9)), 1)[0]
        print(f'{lbl:>22}{p:>18.3f}{"0.741 (теория 0.750)":>16}')


if __name__ == '__main__':
    main()
