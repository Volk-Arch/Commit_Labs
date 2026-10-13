#!/usr/bin/env python3
"""Слепа ли нормированная холономия к плотности ПО ПОСТРОЕНИЮ.

holosource.py: отклика на разрежение нет (+0.149, -0.089, -0.183 при
1.3, 0.7 и 2.1 сигмы, знак перескакивает). Та же картина была у warp
(2.814, 2.808, 2.815).

ПОДОЗРЕНИЕ. Калибровочная фаза есть РАЗНОСТЬ СЧЁТОВ, и я поделил её на
счёт же, чтобы сделать безразмерной. А разрежение меняет плотность, то
есть все счёта разом: числитель и знаменатель уезжают вместе, и
величина к такому источнику слепа ПО ПОСТРОЕНИЮ.

Если так, то НЕНОРМИРОВАННАЯ холономия откликнется сильно — но
тривиально, просто счётом. И тогда вилка та же, что весь день: размерное
откликается даром, безразмерное не откликается вовсе.
"""
import math
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from topo import cone_rel
from holo2 import links_of, cycles_through, holonomy, antisym_causal
from response import warped_diamond

N, T, RAD, SAMP, NCYC, SEEDS = 2600, 2.0, 0.24, 350, 6, 12
LO, HI = RAD, RAD + 0.16

def shell(X, rs):
    R = cone_rel(X); L = links_of(R); U = L | L.T
    A = R.astype(np.float32)
    CP = A @ A.T
    G = antisym_causal((CP - (A.T @ A)).astype(np.float64), R)
    rr = np.linalg.norm(X[:, 1:], axis=1)
    pool = np.flatnonzero((rr >= LO) & (rr < HI) & (U.sum(1) >= 3))
    if len(pool) < 25:
        del R, L, U; return None
    pool = rs.choice(pool, min(SAMP, len(pool)), replace=False)
    raw, nrm, scl = [], [], []
    for v in pool:
        v = int(v)
        cyc = cycles_through(U, v, NCYC, rs)
        if len(cyc) < 2:
            continue
        for c in cyc:
            x = abs(holonomy(c, G))
            sc = np.mean([CP[c[k], c[(k + 1) % len(c)]]
                          for k in range(len(c))])
            sc = max(float(sc), 1.0)
            raw.append(x); nrm.append(x / sc); scl.append(sc)
    del R, L, U
    if not raw:
        return None
    return float(np.mean(raw)), float(np.mean(nrm)), float(np.mean(scl))

print(f'слой {LO:.2f}..{HI:.2f}, засевов {SEEDS}; дырка против '
      f'случайного удаления того же числа')
print(f"{'f':>6}{'удал':>7}{'СЫРАЯ дырка':>14}{'случ.':>10}{'разн.':>9}"
      f"{'сигм':>6}{'НОРМ разн.':>12}{'сигм':>6}{'масштаб дырка':>15}"
      f"{'случ.':>9}")
for f in (1.0, 0.6, 0.3, 0.0):
    rows = []
    for t in range(SEEDS):
        rs = np.random.default_rng(6601000 + 977 * t)
        X0 = warped_diamond(N, T, 0.0, 0.45, rs)
        rr = np.linalg.norm(X0[:, 1:], axis=1)
        ins = np.flatnonzero(rr < RAD)
        drop = ins[rs.random(len(ins)) > f] if len(ins) else ins
        k = len(drop)
        keep = np.ones(len(X0), bool); keep[drop] = False
        r1 = shell(X0[keep], np.random.default_rng(11 + t))
        if k > 0:
            ii = rs.choice(len(X0), len(X0) - k, replace=False)
            r2 = shell(X0[np.sort(ii)], np.random.default_rng(11 + t))
        else:
            r2 = r1
        del X0
        if r1 and r2:
            rows.append((k, r1[0], r2[0], r1[1], r2[1], r1[2], r2[2]))
    if len(rows) < 3:
        print(f'{f:>6.2f}  мало'); continue
    A = np.array(rows, float)
    dr = A[:, 1] - A[:, 2]; dn = A[:, 3] - A[:, 4]
    ser = dr.std(ddof=1) / math.sqrt(len(dr))
    sen = dn.std(ddof=1) / math.sqrt(len(dn))
    print(f'{f:>6.2f}{A[:,0].mean():>7.0f}{A[:,1].mean():>14.1f}'
          f'{A[:,2].mean():>10.1f}{dr.mean():>+9.1f}'
          f'{(abs(dr.mean())/ser if ser > 1e-12 else float("nan")):>6.1f}'
          f'{dn.mean():>+12.4f}'
          f'{(abs(dn.mean())/sen if sen > 1e-12 else float("nan")):>6.1f}'
          f'{A[:,5].mean():>15.1f}{A[:,6].mean():>9.1f}')

print('\n  если СЫРАЯ откликается сильно, а НОРМ нет — слепота по')
print('  построению: числитель и знаменатель уезжают вместе')
