#!/usr/bin/env python3
"""Значимость ПО ЗАСЕВАМ, а не по событиям.

holo2.py дал 8.2 сигмы, но сигмы считались по числу СОБЫТИЙ (10 000), а
события внутри одного мира не независимы. Это та же ошибка, что утром с
перекрывающимися заплатами, где 4.9 сигмы превратились в 1.9.

Честно: частная корреляция считается ВНУТРИ каждого засева, затем
среднее и ошибка среднего ПО ЗАСЕВАМ. Эффективное число наблюдений —
число миров.

Контроли те же: телескопирующаяся фаза (холономия обязана быть нулём) и
перемешанная.
"""
import math
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from topo import diamond, cone_rel
from holo2 import links_of, cycles_through, holonomy, antisym_causal

N, T, SAMP, NCYC, SEEDS = 2200, 2.0, 500, 6, 24

def partial(v, inf, h):
    rv = v - np.polyval(np.polyfit(h, v, 2), h)
    ri = inf - np.polyval(np.polyfit(h, inf, 2), h)
    if rv.std() < 1e-12 or ri.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(rv, ri)[0, 1])

res = {k: [] for k in ('gauge', 'shuf')}
holo_tele = []
for t in range(SEEDS):
    rs = np.random.default_rng(5901000 + 977 * t)
    X = diamond(N, T, rs)
    R = cone_rel(X)
    L = links_of(R)
    U = L | L.T
    A = R.astype(np.float32)
    G = antisym_causal(((A @ A.T) - (A.T @ A)).astype(np.float64), R)
    sz = A.sum(1)
    TE = sz[:, None] - sz[None, :]
    idx = rs.permutation(len(X))
    SH = G[np.ix_(idx, idx)]; SH = 0.5 * (SH - SH.T)
    infl = (L.sum(1) + L.sum(0)) / np.maximum(R.sum(1) + R.sum(0), 1)
    h = np.zeros(len(X), np.int64)
    for i in np.argsort(R.sum(1), kind='stable'):
        i = int(i)
        pre = np.flatnonzero(R[i])
        if len(pre):
            h[i] = h[pre].max() + 1
    deg = U.sum(1)
    pool = np.flatnonzero(deg >= 3)
    if len(pool) < 50:
        del R, L, U, X; continue
    pool = rs.choice(pool, min(SAMP, len(pool)), replace=False)
    vg, vs, vi, vh = [], [], [], []
    for v in pool:
        v = int(v)
        cyc = cycles_through(U, v, NCYC, rs)
        if len(cyc) < 2:
            continue
        vg.append(np.mean([abs(holonomy(c, G)) for c in cyc]))
        vs.append(np.mean([abs(holonomy(c, SH)) for c in cyc]))
        holo_tele.append(np.mean([abs(holonomy(c, TE)) for c in cyc]))
        vi.append(infl[v]); vh.append(float(h[v]))
    del R, L, U, X
    if len(vg) < 60:
        continue
    vg = np.array(vg); vs = np.array(vs)
    vi = np.array(vi); vh = np.array(vh)
    res['gauge'].append(partial(vg, vi, vh))
    res['shuf'].append(partial(vs, vi, vh))

print(f'независимых миров: {len(res["gauge"])}')
print(f'холономия телескопирующейся фазы: '
      f'максимум {max(holo_tele):.2e} (обязана быть 0)')
print()
print(f"{'фаза':>16}{'средняя частная корр.':>24}{'± ошибка':>11}"
      f"{'сигм':>7}{'знак один':>11}")
for k, nm in (('gauge', 'калибровочная'), ('shuf', 'перемешанная')):
    a = np.array([x for x in res[k] if np.isfinite(x)])
    if len(a) < 3:
        print(f'{nm:>16}{"мало":>24}'); continue
    se = a.std(ddof=1) / math.sqrt(len(a))
    same = int((a < 0).sum()) if a.mean() < 0 else int((a > 0).sum())
    print(f'{nm:>16}{a.mean():>+24.4f}{se:>11.4f}'
          f'{abs(a.mean())/max(se,1e-12):>7.1f}{same}/{len(a):>10}')
