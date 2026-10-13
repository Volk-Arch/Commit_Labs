#!/usr/bin/env python3
"""Конусный мир: настоящий причинный порядок вместо покомпонентного.

correlated-scales-doc.md показал, что наше правило верно, а неверной была
независимость шкал: если значения шкал — проекции одной m-мерной точки,
то же правило даёт причинность Минковского. Здесь собран мир на точном
конусе, чтобы пересчитать прежние измерения на настоящем причинном
порядке.

Матрица N x N не строится: интервал между парой определяется фильтром по
координатам за O(N*d), подматрица — только для событий внутри интервала.

Контроль годности: размерность должна выходить 4.00, а спектр
положительным (у покомпонентного порядка было 3.75 и отрицательный).

  python3 coneworld.py          # контроль
"""
import math

import numpy as np

from core import rel_of, dimension, height, spectral, frac_pairs
from chaingrav import chain_len


def sprinkle(N, d, rs, scale=1.0):
    """Засев в причинном ромбе |t| + |x| <= scale."""
    pts = []
    while len(pts) < N:
        c = rs.uniform(-scale, scale, (4 * N, d))
        ok = np.abs(c[:, 0]) + np.linalg.norm(c[:, 1:], axis=1) <= scale
        pts.extend(c[ok][:N - len(pts)])
    return np.array(pts[:N], dtype=np.float64)


def ball4(M, d, r, centre, rs):
    """M событий в шаре радиуса r (во всех d координатах)."""
    v = rs.normal(size=(M, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    rad = r * rs.random(M) ** (1.0 / d)
    return np.asarray(centre) + v * rad[:, None]


def cone_rel(X, sign=None):
    """R[i,j] = True, если j причинно раньше i.

    sign — вектор зарядов; где знаки пары различны, время обращается
    (аналог Фейнмана–Штюкельберга). None — обычный порядок.
    """
    dt = X[:, None, 0] - X[None, :, 0]
    dx = np.linalg.norm(X[:, None, 1:] - X[None, :, 1:], axis=-1)
    if sign is None:
        R = dt > dx
    else:
        same = (np.asarray(sign)[:, None] * np.asarray(sign)[None, :]) > 0
        R = np.where(same, dt > dx, -dt > dx)
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)


def between(X, a, b):
    """Маска событий, причинно лежащих между a и b (a раньше b)."""
    dta = X[:, 0] - X[a, 0]
    dxa = np.linalg.norm(X[:, 1:] - X[a, 1:], axis=1)
    dtb = X[b, 0] - X[:, 0]
    dxb = np.linalg.norm(X[b, 1:] - X[:, 1:], axis=1)
    m = (dta > dxa) & (dtb > dxb)
    m[a] = m[b] = False
    return m


def pick_pairs(X, n_pairs, rs, min_tau=0.6):
    """Пары с достаточным собственным временем между ними."""
    N = len(X)
    out = []
    tries = 0
    while len(out) < n_pairs and tries < 60 * n_pairs:
        tries += 1
        i, j = rs.integers(0, N, 2)
        if i == j:
            continue
        a, b = (int(i), int(j)) if X[i, 0] < X[j, 0] else (int(j), int(i))
        dt = X[b, 0] - X[a, 0]
        dx = np.linalg.norm(X[b, 1:] - X[a, 1:])
        if dt - dx > min_tau:
            out.append((a, b))
    return out


def interval_order(Xbg, ib, a, b, Xadd=None):
    """Подматрица порядка на интервале: [a] + фон внутри + масса + [b]."""
    parts = [Xbg[a][None], Xbg[ib]]
    if Xadd is not None and len(Xadd):
        parts.append(Xadd)
    parts.append(Xbg[b][None])
    return cone_rel(np.vstack(parts)), len(ib)


def links_of_sub(R):
    A = np.asarray(R, dtype=np.float32)
    return R & ((A @ A) == 0)


def main():
    print('КОНТРОЛЬ ГОДНОСТИ КОНУСНОГО МИРА')
    print('  ожидается размерность 4.00 и ПОЛОЖИТЕЛЬНЫЙ спектр')
    print('  (у покомпонентного порядка было 3.75 и отрицательный)')
    print()
    print(f"{'N':>7}{'доля пар':>11}{'размерность':>16}{'высота':>9}{'спектр':>11}")
    for N in (600, 1200, 2400):
        V = []
        for s in range(4):
            rs = np.random.default_rng(100 + 17 * s)
            R = cone_rel(sprinkle(N, 4, rs))
            V.append((frac_pairs(R), dimension(R=R), height(R)[0], spectral(R)))
        V = np.array(V)
        m, sd = V.mean(0), V.std(0) / 2
        print(f'{N:>7}{m[0]:>11.4f}{f"{m[1]:.3f} ± {sd[1]:.3f}":>16}'
              f'{m[2]:>9.1f}{m[3]:>+11.4f}')
    print()
    print('  локальность остаётся точной по той же причине, что и раньше:')
    print('  любое событие цепи от a к b причинно лежит между a и b')


if __name__ == '__main__':
    main()
