#!/usr/bin/env python3
"""Перепроверка вмятины в ПРАВИЛЬНОЙ геометрии.

ПОВОД. weight-doc.md: вмятина от плотной трубки измерена до 13 сигм, её
длина не зависела от размера мира, и она следовала за объектом. Но всё
это измерялось в ящике 4.0 x 2^3, унаследованном из inertia.py, а он НЕ
манифолдоподобен: доля пар 0.5184, размерность 1.952, высота 48 вместо
0.10 / 4.0 / ~9. Измерение было РАЗНОСТНЫМ (трубка минус рассыпанный
контроль, далёкое поле вычтено), поэтому геометрия должна была в
основном сократиться — но это предположение, а не проверка.

Здесь то же измерение в причинном ромбе, где размерность 3.98.

  N = 150*T^4 (плотность 1146), дальнобойность веса при eps=0.2
  составляет 0.427, то есть 0.17 от мира при T=2.5 — лучше, чем 0.9/2.0
  в прежнем ящике.

ПРОВЕРЯЕТСЯ ТО ЖЕ:
  1. есть ли локальная вмятина и какова её значимость;
  2. не зависит ли её длина от размера мира (растущий ромб,
     постоянная плотность);
  3. следует ли она за движущимся объектом (профиль в его системе покоя).

  python3 dentdiam.py
  python3 dentdiam.py --ts 1.5 2.0 2.5 --seeds 10
"""
import argparse
import math
import time

import numpy as np

from weight import diamond, cone_rel, between, per_event


def tube(M, rad, T, rs, v=0.0):
    """Мировая трубка вдоль оси ромба, при v != 0 — движущаяся."""
    u = rs.normal(size=(M, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rad * rs.random(M) ** (1.0 / 3.0)
    X = np.empty((M, 4))
    X[:, 0] = rs.uniform(rad, T - rad, M)
    X[:, 1:] = u * r[:, None]
    X[:, 1] += v * (X[:, 0] - T / 2)
    return X


def scatter_in(M, T, rs):
    """Контроль: те же M событий, рассыпанные по ромбу."""
    return diamond(T, rs, n_at_t1=M / T ** 4)


def boost(X, v, T):
    if v == 0.0:
        return X
    g = 1.0 / math.sqrt(1.0 - v * v)
    Y = X.copy()
    t, x = X[:, 0] - T / 2, X[:, 1]
    Y[:, 0] = g * (t - v * x) + T / 2
    Y[:, 1] = g * (x - v * t)
    return Y


def profile(X, s, M, T, edges, v=0.0):
    """Средний взвешенный вклад ФОНОВЫХ событий по расстоянию до оси."""
    Y = boost(X, v, T)[M:]
    sv = s[M:]
    keep = np.abs(Y[:, 0] - T / 2) < 0.4 * T
    r = np.linalg.norm(Y[:, 1:], axis=1)
    out = np.full(len(edges) - 1, np.nan)
    k = np.digitize(r, edges) - 1
    for b in range(len(out)):
        m = keep & (k == b)
        if m.sum() > 5:
            out[b] = sv[m].mean()
    return out


def run(T, M, rad, eps, seeds, edges, v=0.0):
    acc = {True: [], False: []}
    N = 0
    for t in range(seeds):
        for mk in (True, False):
            rs = np.random.default_rng(141000 + 977 * t)
            bg = diamond(T, rs)
            ob = tube(M, rad, T, rs, v) if mk else scatter_in(M, T, rs)
            X = np.vstack([ob, bg])
            R = cone_rel(X)
            btw = between(R)
            s, _ = per_event(R, btw, eps)
            acc[mk].append(profile(X, s, len(ob), T, edges, v))
            N = len(X)
            del R, btw, s, X
    A = {mk: np.array(acc[mk], float) for mk in acc}
    m = {mk: np.nanmean(A[mk], 0) for mk in A}
    e = {mk: np.nanstd(A[mk], 0) / math.sqrt(seeds) for mk in A}
    d = m[True] - m[False]
    er = np.hypot(e[True], e[False])
    far = np.nanmean(d[-2:])
    return d - far, er, N


def main():
    ap = argparse.ArgumentParser(description='Вмятина в ромбе')
    ap.add_argument('--ts', type=float, nargs='+', default=[1.5, 2.0, 2.5])
    ap.add_argument('--m', type=int, default=600)
    ap.add_argument('--rad', type=float, default=0.02)
    ap.add_argument('--eps', type=float, default=0.2)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--vs', type=float, nargs='+', default=[0.0, 0.3, 0.5])
    ap.add_argument('--fixline', action='store_true',
                    help='держать линейную плотность трубки: M ~ T')
    a = ap.parse_args()

    t0 = time.time()
    edges = np.array([0.0, 0.15, 0.25, 0.4, 0.6, 0.85, 1.2])
    print(f'причинный ромб, N = 150*T^4, трубка {a.m} событий радиуса '
          f'{a.rad}, eps={a.eps}')
    print(f'дальнобойность веса ~0.43; засевов {a.seeds}')
    print('(в прежнем ящике мир не был манифолдоподобен: '
          'доля 0.52, размерность 1.95)')
    print()
    print('1-2. ВМЯТИНА И ЕЁ ДЛИНА ПРОТИВ РАЗМЕРА МИРА '
          '(далёкое поле вычтено)')
    print(f"{'T':>5}{'N':>7}" + ''.join(f'{e:>9.2f}' for e in edges[1:-1]))
    for T in a.ts:
        M = int(round(a.m * T / max(a.ts))) if a.fixline else a.m
        d, er, N = run(T, M, a.rad, a.eps, a.seeds, edges)
        n = len(edges) - 2
        print(f'{T:>5.1f}{N:>7}' + ''.join(f'{v:>9.4f}' for v in d[:n])
              + f'   M={M}')
        print(f"{'± ошибка':>12}"
              + ''.join(f'{x:>9.4f}' for x in er[:n]))
        print(f"{'значим.':>12}"
              + ''.join(f'{(abs(v)/x if x > 1e-5 else float("nan")):>9.1f}'
                        for v, x in zip(d[:n], er[:n])), flush=True)

    print()
    print(f'3. СЛЕДУЕТ ЛИ ЗА ОБЪЕКТОМ (T={a.ts[-1]}, профиль в его '
          f'системе покоя)')
    print(f"{'v':>5}{'N':>7}" + ''.join(f'{e:>9.2f}' for e in edges[1:-1]))
    for v in a.vs:
        d, er, N = run(a.ts[-1], a.m, a.rad, a.eps, a.seeds, edges, v)
        n = len(edges) - 2
        print(f'{v:>5.1f}{N:>7}' + ''.join(f'{x:>9.4f}' for x in d[:n]))
        print(f"{'± ошибка':>12}"
              + ''.join(f'{y:>9.4f}' for y in er[:n]))
        print(f"{'значим.':>12}"
              + ''.join(f'{(abs(x)/y if y > 1e-5 else float("nan")):>9.1f}'
                        for x, y in zip(d[:n], er[:n])), flush=True)
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
