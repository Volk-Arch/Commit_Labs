#!/usr/bin/env python3
"""УТОЧНЕНИЕ: 0.0661 — это пол или край моей сетки?

wscan.py просканировал 1734 точки гиперплоскости геометрических весов
(`sum w_m Gamma(m+1/2)/m! = 0`) и не нашёл ни одной, где слоистый
сектор гибнет: минимум c упал с 0.1167 (веса БД) до 0.0661, но
остался положительным. ОДНАКО лучшая точка стояла на КРАЮ сетки, и
вывод делать нельзя, пока не ясно, пол это или граница обзора.

Здесь три вещи:
  1. локальный спуск (Нелдер–Мид) с лучшей точки, без сетки;
  2. многократный пуск из случайных начал — не застрял ли спуск;
  3. РАСШИРЕНИЕ пространства: веса до w_6 вместо w_3. Геометрическое
     условие остаётся одним линейным, а свободы становится больше.
     Если и там пол положителен — вывод прочный.

КОНТРОЛЬ: якорь на весах БД печатается в каждом прогоне.
"""
import math

import numpy as np
from scipy.optimize import minimize

MAXM = 6
GAM = np.array([math.gamma(m + 0.5) / math.factorial(m)
                for m in range(MAXM + 1)])
BD = np.array([-1.0, 9.0, -16.0, 8.0] + [0.0] * (MAXM - 3))
MS = np.arange(1, 41)
DS = np.arange(1, 41)
CMT = {t: np.array([math.comb(int(m), t) if m >= t else 0
                    for m in MS], float) for t in range(1, MAXM + 1)}


def maxc(w, th, kmax):
    u = np.exp(1j * th * w[0])
    base = (1.0 + u) ** MS
    best = -np.inf
    for d in DS:
        tot = base.copy()
        for t in range(1, kmax + 1):
            if abs(w[t]) < 1e-12:
                continue
            tot = tot + CMT[t] * u ** t * (
                np.exp(1j * th * w[t] * d) - 1.0)
        c = np.log(np.maximum(np.abs(tot), 1e-300)) / (4.0 * (MS + d))
        best = max(best, float(c.max()))
    return best


def objective(free, kmax, ths, half=3):
    """free = w1..wkmax; w0 вынужден условием; нормировка max|w| = 16."""
    w = np.zeros(MAXM + 1)
    w[1:kmax + 1] = free
    w[0] = -float(free @ GAM[1:kmax + 1])
    s = np.abs(w).max()
    if s < 1e-9:
        return 1e3
    w = w * (16.0 / s)
    c = np.array([maxc(w, t, kmax) for t in ths])
    k = 2 * half + 1
    if len(c) < k:
        return float(c.min())
    sl = np.lib.stride_tricks.sliding_window_view(c, k).max(axis=1)
    return float(sl.min())


def main():
    print(__doc__)
    ths = np.linspace(0.05, 3.0, 240)
    print(f'ЯКОРЬ, веса БД: геом. условие {float(BD @ GAM):.2e}, '
          f'c = {objective(BD[1:4], 3, ths):.4f} (архив 0.1167)')

    for kmax in (3, 6):
        print(f'\n=== свобода до w_{kmax} ===')
        res = []
        starts = [np.array([-16.0, -16.0, -3.2] + [0.0] * (kmax - 3))]
        rs = np.random.default_rng(4242 + kmax)
        for _ in range(7):
            starts.append(rs.uniform(-16, 16, kmax))
        for s0 in starts:
            r = minimize(objective, s0, args=(kmax, ths),
                         method='Nelder-Mead',
                         options=dict(maxiter=900, xatol=1e-3,
                                      fatol=1e-5))
            res.append((float(r.fun), r.x.copy()))
        res.sort(key=lambda z: z[0])
        print(f"{'c':>9}   веса w1..w{kmax} (нормировка внутри)")
        for v, x in res[:4]:
            print(f'{v:>9.4f}   ' + ', '.join(f'{u:+.2f}' for u in x))
        print(f'  лучшее {res[0][0]:.4f}; пусков с c <= 0: '
              f'{sum(1 for v, _ in res if v <= 0)}/{len(res)}')
        w = np.zeros(MAXM + 1)
        w[1:kmax + 1] = res[0][1]
        w[0] = -float(res[0][1] @ GAM[1:kmax + 1])
        print('  полный вес (w0..): '
              + ', '.join(f'{v:+.3f}' for v in w[:kmax + 1])
              + f'  | геом. условие {float(w @ GAM):.1e}')

    print('\nЧТЕНИЕ: если спуск из разных начал упирается в один')
    print('положительный пол, и расширение до w_6 его не пробивает —')
    print('значит гашение фаз закрыто не только для весов БД, а для')
    print('ВСЕГО класса геометрических действий этого вида.')


if __name__ == '__main__':
    main()
