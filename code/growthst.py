#!/usr/bin/env python3
"""p*(N) ПРЯМЫМ ЗАМЕРОМ и вес настоящего четырёхмерного засева.

Две входные величины для сравнения секторов под мерой роста, обе
МЕРЕНЫЕ, а не модельные:

  1. `p*(N)`: точка, где у транзитивной перколяции размерность
     Мирхейма-Мейера равна четырём, то есть доля отношений r = 2c_2(4)
     = 0.1. Прежнее значение 0.09 подогнано при N <= 7 со сносом 2.5%
     на шаг — экстраполировать такое нельзя, надо мерить;
  2. `R` и `L` настоящего засева в четырёхмерный диамант: число
     отношений и число СВЯЗЕЙ (пар с пустым интервалом). Отсюда
     ln Pr = L ln p + (M - R) ln(1-p) для пространства-времени.

КОНТРОЛИ:
  * замыкание перколяции битовыми множествами против numpy-матрицы;
  * у засева доля r обязана дать размерность около 4 САМА (это
    проверка засева, а не подгонка);
  * связи засева: L/N должно расти как N^{1/2} при d=4 (известная
    асимптотика N^{2-2/d}) — проверяется наклоном.

  python3 growthst.py
"""
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma


def c2(d):
    return gamma(d + 1) * gamma(d / 2) * gamma(d) / (
        4.0 * gamma(d) * gamma(3 * d / 2))


def mmdim(r):
    try:
        return brentq(lambda d: 2 * c2(d) - r, 0.4, 40.0)
    except Exception:
        return float('nan')


R_TARGET = 2 * c2(4.0)


def perc_r(n, p, rs):
    """Доля отношений транзитивной перколяции, замыкание битсетами."""
    reach = [0] * n
    tot = 0
    for i in range(n - 1, -1, -1):
        row = rs.random(n - i - 1) < p
        m = 0
        for off, has in enumerate(row):
            if has:
                j = i + 1 + off
                m |= (1 << j) | reach[j]
        reach[i] = m
        tot += bin(m).count('1')
    return tot / (n * (n - 1) / 2)


def perc_r_mat(n, p, rs):
    """КОНТРОЛЬ: то же через numpy-замыкание."""
    A = np.triu(rs.random((n, n)) < p, 1)
    Rc = A.copy()
    while True:
        nxt = Rc | ((Rc.astype(np.int32) @ Rc.astype(np.int32)) > 0)
        if (nxt == Rc).all():
            break
        Rc = nxt
    return Rc.sum() / (n * (n - 1) / 2)


def pstar(n, rs, reps=3):
    def f(p):
        return np.mean([perc_r(n, p, rs) for _ in range(reps)]) - R_TARGET
    return brentq(f, 1e-4, 0.5, xtol=1e-5)


def sprinkle(n, rs):
    """Засев в четырёхмерный causal diamond, отбраковкой."""
    T = 1.0
    pts = []
    while len(pts) < n:
        k = int((n - len(pts)) * 4.5) + 16
        t = rs.random(k) * T
        v = rs.normal(size=(k, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        rad = (rs.random(k) ** (1 / 3)) * (T / 2)
        x = v * rad[:, None]
        ok = rad < np.minimum(t, T - t)
        for i in np.nonzero(ok)[0]:
            pts.append((t[i], *x[i]))
    P = np.array(pts[:n])
    return P


def rel_links(P):
    """Матрица причинности, число отношений и число связей."""
    n = len(P)
    dt = P[:, 0][None, :] - P[:, 0][:, None]
    dx2 = ((P[:, 1:][None, :, :] - P[:, 1:][:, None, :]) ** 2).sum(-1)
    Rc = (dt > 0) & (dt ** 2 > dx2)
    two = (Rc.astype(np.float32) @ Rc.astype(np.float32)) > 0.5
    # контроль транзитивности: Rc@Rc должно лежать в Rc
    assert not (two & ~Rc).any(), 'причинность не транзитивна'
    L = int((Rc & ~two).sum())
    return int(Rc.sum()), L


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(20261009)
    print(f'цель по доле отношений для d=4: r = {R_TARGET:.6f}')
    print()
    print('КОНТРОЛЬ: битсетное замыкание против numpy')
    for n, p in ((60, 0.08), (120, 0.05)):
        a = np.mean([perc_r(n, p, rs) for _ in range(40)])
        b = np.mean([perc_r_mat(n, p, rs) for _ in range(40)])
        print(f'  N={n}, p={p}: r битсеты {a:.5f}, numpy {b:.5f}')
    print()
    print('p*(N) ПРЯМЫМ ЗАМЕРОМ (перколяция, r = 0.1)')
    print(f"{'N':>6}{'p*':>10}{'p* * N':>10}")
    ps = {}
    for n in (10, 20, 40, 80, 160, 320, 640):
        p = pstar(n, rs, reps=4 if n < 200 else 2)
        ps[n] = p
        print(f'{n:>6}{p:>10.5f}{p*n:>10.3f}', flush=True)
    print()
    print('ЧЕТЫРЁХМЕРНЫЙ ЗАСЕВ: отношения и связи')
    print(f"{'N':>6}{'r засева':>11}{'d Мирхейма':>12}{'L':>10}"
          f"{'L/N^1.5':>10}")
    sp = {}
    for n in (50, 100, 200, 400, 800):
        rr, ll = [], []
        for _ in range(3 if n <= 400 else 1):
            P = sprinkle(n, rs)
            R, L = rel_links(P)
            rr.append(R)
            ll.append(L)
        R = float(np.mean(rr))
        L = float(np.mean(ll))
        M = n * (n - 1) / 2
        sp[n] = (R, L, M)
        print(f'{n:>6}{R/M:>11.5f}{mmdim(R/M):>12.3f}{L:>10.1f}'
              f'{L/n**1.5:>10.4f}', flush=True)
    np.save('/home/claude/data/growth_inputs.npy',
            np.array([[n, ps.get(n, np.nan)] for n in sorted(ps)],
                     dtype=float))
    import json
    json.dump({'pstar': {str(k): v for k, v in ps.items()},
               'sprinkle': {str(k): list(v) for k, v in sp.items()}},
              open('/home/claude/data/growth_inputs.json', 'w'), indent=1)
    print()
    print('записано в /home/claude/data/growth_inputs.json')


if __name__ == '__main__':
    main()
