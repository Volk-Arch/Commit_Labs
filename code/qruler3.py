#!/usr/bin/env python3
"""ПОКАЗАТЕЛЬ БЮДЖЕТА ПЕРВОЙ ЛИНЕЙКИ на больших V.

`qruler2.py` дал бюджет 8.6...12.0 при V_общ 25...600, то есть
~V^0.09 — почти плато, но показатель с погрешностью порядка 0.04 на
24-кратном размахе и 108 парах в верхней корзине. Всё следствие
(«потолок не снимается плотностью») висит на этом числе, поэтому его
надо добить.

Здесь N до 8000 на засев, матрица отношений строится блоками, чтобы не
держать N^2 чисел с плавающей точкой.

  python3 qruler3.py
"""
import math

import numpy as np


def sprinkle4(n, rs, T=1.0):
    out = []
    got = 0
    while got < n:
        m = int((n - got) * 5) + 256
        t = rs.random(m) * T
        v = rs.normal(size=(m, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        rad = (rs.random(m) ** (1.0 / 3.0)) * (T / 2)
        ok = rad < np.minimum(t, T - t)
        if ok.any():
            out.append(np.column_stack([t[ok], (v * rad[:, None])[ok]]))
            got += int(ok.sum())
    return np.vstack(out)[:n].astype(np.float32)


def rel_blocks(P, blk=400):
    n = len(P)
    Rc = np.zeros((n, n), dtype=bool)
    t = P[:, 0]
    X = P[:, 1:]
    for i0 in range(0, n, blk):
        i1 = min(i0 + blk, n)
        dt = t[None, :] - t[i0:i1, None]
        d = X[None, :, :] - X[i0:i1, None, :]
        dx2 = (d * d).sum(-1)
        Rc[i0:i1] = (dt > 0) & (dt * dt > dx2)
    return Rc


def harvest(n, rs, npair, agew=0.03, T=1.0):
    P = sprinkle4(n, rs, T)
    Rc = rel_blocks(P)
    Vp = Rc.sum(axis=0)
    tau = np.sqrt(np.maximum(P[:, 0].astype(np.float64) ** 2
                             - (P[:, 1:].astype(np.float64) ** 2).sum(1),
                             0.0))
    rows = []
    tries = 0
    lim = npair * 300
    while len(rows) < npair and tries < lim:
        tries += 1
        a, b = int(rs.integers(n)), int(rs.integers(n))
        if a == b or Rc[a, b] or Rc[b, a]:
            continue
        if Vp[a] < 40 or Vp[b] < 40:
            continue
        if abs(tau[a] / max(tau[b], 1e-9) - 1.0) > agew:
            continue
        dt = float(P[b, 0]) - float(P[a, 0])
        dx2 = float(((P[b, 1:] - P[a, 1:]) ** 2).sum())
        sep = math.sqrt(max(dx2 - dt * dt, 0.0))
        tb = 0.5 * (tau[a] + tau[b])
        vcp = int((Rc[:, a] & Rc[:, b]).sum())
        u = vcp / (0.5 * (Vp[a] + Vp[b]))
        rows.append((sep / tb, u, vcp))
    return np.array(rows)


def budget(D, lo, hi, nq=9):
    s = (D[:, 2] >= lo) & (D[:, 2] < hi)
    if s.sum() < 120:
        return None
    xs, us = D[s, 0], D[s, 1]
    qs = np.quantile(us, np.linspace(0, 1, nq))
    res = []
    for j in range(len(qs) - 1):
        m = (us >= qs[j]) & (us <= qs[j + 1])
        if m.sum() < 15:
            continue
        res.append(xs[m] - np.median(xs[m]))
    if not res:
        return None
    rr = np.concatenate(res)
    cond = float(np.std(rr, ddof=1))
    return (float(D[s, 2].mean()), float(xs.mean()),
            cond, float(xs.mean()) / cond, int(s.sum()))


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(99001)
    parts = []
    for n, k, npr in ((2000, 4, 40000), (4000, 3, 40000), (8000, 2, 40000)):
        for _ in range(k):
            parts.append(harvest(n, rs, npr))
        print(f'  собрано при N = {n}: пар '
              f'{sum(len(p) for p in parts[-k:])}', flush=True)
    D = np.vstack(parts)
    print(f'всего пар {len(D)}, V_общ до {int(D[:,2].max())}')
    print()
    print(f"{'V_общ':>14}{'пар':>8}{'<x>':>8}{'усл. разброс':>14}"
          f"{'бюджет':>9}")
    rows = []
    for lo, hi in ((40, 100), (100, 250), (250, 600), (600, 1500),
                   (1500, 4000), (4000, 12000)):
        r = budget(D, lo, hi)
        if r is None:
            continue
        rows.append(r)
        print(f'{f"{lo}-{hi}":>14}{r[4]:>8}{r[1]:>8.3f}{r[2]:>14.4f}'
              f'{r[3]:>9.2f}', flush=True)
    R = np.array([(r[0], r[3], r[4]) for r in rows])
    err = R[:, 1] / np.sqrt(2 * (R[:, 2] - 1))
    x, y = np.log(R[:, 0]), np.log(R[:, 1])
    w = 1.0 / (err / R[:, 1]) ** 2
    A = np.vstack([x, np.ones_like(x)]).T
    W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A)
    be = cov @ (A.T @ W @ y)
    se = np.sqrt(np.diag(cov))
    print()
    print(f'ПОКАЗАТЕЛЬ: бюджет ~ V^a,  a = {be[0]:.4f} ± {se[0]:.4f}, '
          f'коэффициент {math.exp(be[1]):.2f}')
    for cand, lab in ((0.0, 'плато'), (0.25, 'V^1/4'), (0.5, 'V^1/2')):
        print(f'  против {lab:>7}: отличие {abs(be[0]-cand)/se[0]:.1f} сигмы')
    res = (y - A @ be) / (err / R[:, 1])
    print('  остатки (сигм):', ', '.join(f'{v:+.1f}' for v in res))
    print()
    print('ЭКСТРАПОЛЯЦИЯ по замеренному показателю:')
    for V in (1e6, 1e12, 1e60):
        print(f'  V = 1e{int(math.log10(V)):<3}: бюджет '
              f'{math.exp(be[1])*V**be[0]:.3g} радиан = '
              f'{math.exp(be[1])*V**be[0]/(2*math.pi):.3g} оборотов')


if __name__ == '__main__':
    main()
