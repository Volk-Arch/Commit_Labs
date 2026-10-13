#!/usr/bin/env python3
"""УТОЧНЕНИЕ, векторизованное: 0.0661 — пол или край сетки?

wscan.py: 1734 точки гиперплоскости геометрических весов, ни одной с
гибелью сектора; минимум упал с 0.1167 (БД) до 0.0661, но лучшая точка
стояла на КРАЮ сетки. Здесь локальный спуск без сетки, из многих
начал, и расширение свободы до w_6.

Вся сетка (th, m, d) считается одной операцией — прошлая версия
оставляла цикл по d внутри цикла по th и не уложилась.

КОНТРОЛЬ: якорь на весах БД в каждом прогоне (архив: 0.1167).
"""
import math

import numpy as np
from scipy.optimize import minimize

MAXM = 6
GAM = np.array([math.gamma(m + 0.5) / math.factorial(m)
                for m in range(MAXM + 1)])
BD = np.array([-1.0, 9.0, -16.0, 8.0] + [0.0] * (MAXM - 3))
MS = np.arange(1, 36)
DS = np.arange(1, 36)
CM = np.stack([np.array([math.comb(int(m), t) if m >= t else 0
                         for m in MS], float)
               for t in range(1, MAXM + 1)])          # (t, m)
MD = 4.0 * (MS[:, None] + DS[None, :])                # (m, d)


def curve(w, ths):
    """max по (m,d) от c для каждого th — одной векторной операцией."""
    th = ths[:, None, None]
    u = np.exp(1j * ths * w[0])                       # (T,)
    tot = ((1.0 + u)[:, None] ** MS[None, :])[:, :, None] \
        * np.ones((1, 1, len(DS)))
    for t in range(1, MAXM + 1):
        if abs(w[t]) < 1e-12:
            continue
        ph = np.exp(1j * th * w[t] * DS[None, None, :]) - 1.0
        tot = tot + (CM[t - 1][None, :, None]
                     * (u ** t)[:, None, None] * ph)
    c = np.log(np.maximum(np.abs(tot), 1e-300)) / MD[None, :, :]
    return c.reshape(len(ths), -1).max(axis=1)


def norm_w(free, kmax):
    w = np.zeros(MAXM + 1)
    w[1:kmax + 1] = free
    w[0] = -float(free @ GAM[1:kmax + 1])
    s = np.abs(w).max()
    return (w * (16.0 / s)) if s > 1e-9 else None


def objective(free, kmax, ths, half=3):
    w = norm_w(free, kmax)
    if w is None:
        return 1e3
    c = curve(w, ths)
    k = 2 * half + 1
    sl = np.lib.stride_tricks.sliding_window_view(c, k).max(axis=1)
    return float(sl.min())


def main():
    print(__doc__)
    ths = np.linspace(0.05, 3.0, 300)
    print(f'ЯКОРЬ, веса БД: геом. условие {float(BD @ GAM):.2e}, '
          f'c = {objective(BD[1:4], 3, ths):.4f}  (архив 0.1167)')

    for kmax in (3, 4, 6):
        rs = np.random.default_rng(900 + kmax)
        starts = [np.array([-16.0, -16.0, -3.2] + [0.0] * (kmax - 3))]
        starts += [rs.uniform(-16, 16, kmax) for _ in range(9)]
        res = []
        for s0 in starts:
            r = minimize(objective, s0, args=(kmax, ths),
                         method='Nelder-Mead',
                         options=dict(maxiter=1200, xatol=1e-3,
                                      fatol=1e-6))
            res.append((float(r.fun), r.x.copy()))
        res.sort(key=lambda z: z[0])
        print(f'\n=== свобода до w_{kmax} ({len(starts)} пусков) ===')
        for v, x in res[:3]:
            w = norm_w(x, kmax)
            print(f'  c = {v:>8.4f}   вес '
                  + ', '.join(f'{u:+.2f}' for u in w[:kmax + 1])
                  + f'  | геом. {float(w @ GAM):.0e}')
        print(f'  лучшее {res[0][0]:.4f}; пусков с c <= 0: '
              f'{sum(1 for v, _ in res if v <= 0)}/{len(res)}; '
              f'разброс лучших трёх '
              f'{res[0][0]:.4f}…{res[2][0]:.4f}')

    print('\nЧТЕНИЕ: один положительный пол из разных начал и при')
    print('расширении свободы = гашение фаз закрыто для ВСЕГО класса')
    print('геометрических действий этого вида, а не только для БД.')


if __name__ == '__main__':
    main()
