#!/usr/bin/env python3
"""Стена шума на золотой решётке.

Тот же оператор БД 1+1, что в fastwave/noisewall (веса 1,−2,1, c_self=−1/2,
ρ=2N, точки в единичном квадрате световых координат), те же функции
t², x², τ²=t²−x² и то же ядро (0.3<t<0.7, |x|<0.15). На засеве разброс
по элементам растёт как N^1.001. Вопрос: что на золотой решётке.
"""
import sys, math, time, numpy as np
sys.path.insert(0, '/home/claude')
from fastwave import bop_at, CSELF, tx
from goldlat import gold_points

def lattice_uv(Uside):
    u, v = gold_points(float(Uside))
    return np.stack([u / Uside, v / Uside], 1)

def evaluate(P, npts, rs):
    U, V = P[:, 0], P[:, 1]; n = len(P); t, x = tx(P)
    order_u = np.argsort(U); pos = np.empty(n, int); pos[order_u] = np.arange(n)
    ok = np.nonzero((t > 0.3) & (t < 0.7) & (np.abs(x) < 0.15))[0]
    pick = ok if len(ok) <= npts else rs.choice(ok, npts, replace=False)
    rho = 2.0 * n
    fs = {'t²': t * t, 'x²': x * x, 'τ²': t * t - x * x}
    out = {k: [] for k in fs}
    for idx in pick:
        ys, ws = bop_at(U, V, idx, order_u, pos)
        for k, f in fs.items():
            s = float((ws * f[ys]).sum()) if len(ys) else 0.0
            out[k].append(rho * (CSELF * f[idx] + s))
    return n, len(ok), {k: np.array(v) for k, v in out.items()}

rs = np.random.default_rng(5)
print(f"{'':10}{'N':>8}{'ядро':>7} |" + ''.join(f"{k+' среднее':>14}{'σ':>9} |" for k in ('t²', 'x²', 'τ²')))
t0 = time.time()
rows = []
for side in (22, 40, 70, 125, 220, 400):
    P = lattice_uv(side)
    n, nc, out = evaluate(P, 400, rs)
    Q = rs.random((n, 2))
    n2, nc2, out2 = evaluate(Q, 400, rs)
    for lab, nn, cc, o in (('золотая', n, nc, out), ('засев', n2, nc2, out2)):
        print(f'{lab:10}{nn:>8}{cc:>7} |' + ''.join(f'{o[k].mean():>14.3f}{o[k].std():>9.3f} |' for k in ('t²', 'x²', 'τ²')))
    rows.append((n, out['t²'].std(), out2['t²'].std()))
    print(f'   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
ns = np.array([r[0] for r in rows]); sg = np.array([r[1] for r in rows]); sp = np.array([r[2] for r in rows])
print(f'\nпоказатель σ(N) для t²:  золотая {np.polyfit(np.log(ns), np.log(sg), 1)[0]:+.3f}   засев {np.polyfit(np.log(ns), np.log(sp), 1)[0]:+.3f}')
