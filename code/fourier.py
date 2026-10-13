#!/usr/bin/env python3
"""БЕЛЫЙ ЛИ ШУМ ОПЕРАТОРА: можно ли его отфильтровать.

`noisewall-doc.md`: разброс на элемент растёт как N^1.001, нужное число
точек ~N^2, доступное ~N. Фильтр (Фурье, размазка, любое сглаживание)
помогает только если шум и сигнал живут в РАЗНЫХ частотах.

Если шум БЕЛЫЙ (независим между элементами), усреднение по M точкам даёт
sigma/sqrt(M) = N/sqrt(N) = sqrt(N) - растёт, и это ровно измеренное.
Фильтр тогда бессилен: усреднение И ЕСТЬ оптимальный фильтр для белого
шума.

Если шум ДЛИННО-КОРРЕЛИРОВАН, усреднение неоптимально, но тогда шум
сидит на тех же низких частотах, что сигнал, и разделить его тоже
нельзя. Повод подозревать второе: связи ДЛИННЫЕ - 70-80% проходят через
срединный срез (`horizon.py`).

Различить можно прямо: корреляция шума между точками как функция их
расстояния. Мера расстояния - собственное время |dt^2 - dx^2|^(1/2) для
причинных пар и пространственное для остальных.

ЧТО РАЗЛИЧАЕТ ИСХОДЫ:
  * корреляция падает к нулю на масштабе дискретности => шум белый,
    усреднение оптимально, фильтр ничего не даст (и это объясняет, почему
    размазка дала лишь 3.6x);
  * корреляция держится на больших расстояниях => шум низкочастотный,
    то есть неотделим от сигнала по частоте.
"""
import math

import numpy as np

from fastwave import sprinkle_uv, tx, bop_at, CSELF

N = 20000
NPTS = 1200


def field(n, rs, npts):
    P = sprinkle_uv(n, rs)
    U, V = P[:, 0], P[:, 1]
    t, x = tx(P)
    order_u = np.argsort(U)
    pos = np.empty(n, int)
    pos[order_u] = np.arange(n)
    ok = np.nonzero((t > 0.25) & (t < 0.75) & (np.abs(x) < 0.2))[0]
    pick = rs.choice(ok, size=min(npts, len(ok)), replace=False)
    rho = 2.0 * n
    f = x * x
    vals = np.empty(len(pick))
    for i, idx in enumerate(pick):
        ys, ws = bop_at(U, V, idx, order_u, pos)
        s = float((ws * f[ys]).sum()) if len(ys) else 0.0
        vals[i] = rho * (CSELF * f[idx] + s)
    return t[pick], x[pick], vals


def main():
    print(__doc__)
    rs = np.random.default_rng(314159)
    ell = 1.0 / math.sqrt(2.0 * N)
    print(f'N={N}, масштаб дискретности l = 1/sqrt(rho) = {ell:.5f}')
    T, X, B = field(N, rs, NPTS)
    B = B - B.mean()
    sd = B.std()
    print(f'разброс шума {sd:.1f}, точек {len(B)}')

    dt = T[None, :] - T[:, None]
    dx = X[None, :] - X[:, None]
    s2 = dt * dt - dx * dx
    d = np.where(s2 > 0, np.sqrt(np.abs(s2)), -np.sqrt(np.abs(s2)))
    C = np.outer(B, B) / (sd * sd)
    iu = np.triu_indices(len(B), 1)
    dd, cc = d[iu], C[iu]

    print(f"\n{'интервал (в l)':>18}{'пар':>10}{'корреляция':>13}")
    edges = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 1e9]) * ell
    for a, b in zip(edges[:-1], edges[1:]):
        m = (np.abs(dd) >= a) & (np.abs(dd) < b)
        if m.sum() < 50:
            continue
        lo = a / ell
        hi = b / ell if b < 1e8 else float('inf')
        print(f'{f"{lo:.0f} … {hi:.0f}":>18}{int(m.sum()):>10}'
              f'{float(cc[m].mean()):>13.4f}', flush=True)
    print('\nпричинные против пространственноподобных:')
    for lab, m in (('причинные', dd > 0), ('пространств.', dd < 0)):
        print(f'  {lab:>14}: пар {int(m.sum()):>8}, '
              f'корреляция {float(cc[m].mean()):+.4f}')


if __name__ == '__main__':
    main()
