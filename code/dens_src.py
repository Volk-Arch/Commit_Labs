#!/usr/bin/env python3
"""Второе и третье: растёт ли эффект с плотностью, и порождает ли его вещество.

ПЕРВОЕ (отклик) ОТРИЦАТЕЛЬНО: нормированная холономия не отвечает на
наведённое искривление (0.1, 0.0 и 2.5 сигмы в обратную сторону), тогда
как признак кривизны проехал 0.250 -> 0.131. Значит корреляция 7.5 сигмы
уравнением поля не является.

ВТОРОЕ. Если корреляция не растёт с плотностью, она скорее
конечноразмерный артефакт, чем физика. Считаем частную корреляцию
(холономия против доли прямых при снятой высоте) по независимым мирам
при разных N и T, то есть при разной плотности rho = N / объём ромба.

ТРЕТЬЕ. Источник. Если ничто наведённое холономию не создаёт, может,
её создаёт вещество — плотная мировая линия. Меряем нормированную
холономию вблизи подсаженной линии против мира без неё, ПАРНО по засеву.
"""
import math
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from topo import cone_rel
from holo2 import links_of, cycles_through, holonomy, antisym_causal
from response import warped_diamond

def prep(X, rs):
    R = cone_rel(X); L = links_of(R); U = L | L.T
    A = R.astype(np.float32)
    CP = A @ A.T
    G = antisym_causal((CP - (A.T @ A)).astype(np.float64), R)
    infl = (L.sum(1) + L.sum(0)) / np.maximum(R.sum(1) + R.sum(0), 1)
    h = np.zeros(len(X), np.int64)
    for i in np.argsort(R.sum(1), kind='stable'):
        i = int(i)
        pre = np.flatnonzero(R[i])
        if len(pre):
            h[i] = h[pre].max() + 1
    return R, U, G, CP, infl, h

def sample(X, U, G, CP, infl, h, pool, samp, ncyc, rs, norm=False):
    pool = rs.choice(pool, min(samp, len(pool)), replace=False)
    hv, iv, hh = [], [], []
    for v in pool:
        v = int(v)
        cyc = cycles_through(U, v, ncyc, rs)
        if len(cyc) < 2:
            continue
        vals = []
        for c in cyc:
            x = abs(holonomy(c, G))
            if norm:
                sc = np.mean([CP[c[k], c[(k + 1) % len(c)]]
                              for k in range(len(c))])
                x = x / max(float(sc), 1.0)
            vals.append(x)
        hv.append(np.mean(vals)); iv.append(infl[v]); hh.append(float(h[v]))
    return np.array(hv), np.array(iv), np.array(hh)

def partial(v, inf, h):
    rv = v - np.polyval(np.polyfit(h, v, 2), h)
    ri = inf - np.polyval(np.polyfit(h, inf, 2), h)
    if rv.std() < 1e-12 or ri.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(rv, ri)[0, 1])

print('ВТОРОЕ: растёт ли частная корреляция с плотностью')
print(f"{'N':>7}{'T':>5}{'плотность':>11}{'частная корр.':>15}{'±':>9}"
      f"{'сигм':>7}{'миров':>7}")
for N, T in ((1000, 2.0), (2200, 2.0), (4000, 2.0), (2200, 1.5)):
    vals = []
    for t in range(12):
        rs = np.random.default_rng(6301000 + 977 * t)
        X = warped_diamond(N, T, 0.0, 0.45, rs)
        R, U, G, CP, infl, h = prep(X, rs)
        pool = np.flatnonzero(U.sum(1) >= 3)
        if len(pool) < 60:
            del R, U, X; continue
        hv, iv, hh = sample(X, U, G, CP, infl, h, pool, 400, 6, rs)
        del R, U, X
        if len(hv) > 60:
            vals.append(partial(hv, iv, hh))
    a = np.array([x for x in vals if np.isfinite(x)])
    if len(a) < 3:
        print(f'{N:>7}{T:>5.1f}  мало'); continue
    se = a.std(ddof=1) / math.sqrt(len(a))
    rho = N / (math.pi / 24 * T ** 4)
    print(f'{N:>7}{T:>5.1f}{rho:>11.0f}{a.mean():>+15.4f}{se:>9.4f}'
          f'{abs(a.mean())/max(se,1e-12):>7.1f}{len(a):>7}')

print()
print('ТРЕТЬЕ: порождает ли холономию ВЕЩЕСТВО (плотная мировая линия)')
print(f"{'M в линии':>11}{'холон. у линии':>16}{'без линии':>12}"
      f"{'парно':>10}{'сигм':>7}")
N0, T, SIG = 2200, 2.0, 0.25
for M in (0, 40, 120, 300):
    dif = []
    for t in range(10):
        rs = np.random.default_rng(6401000 + 977 * t)
        bg = warped_diamond(N0, T, 0.0, 0.45, rs)
        c = np.zeros(3)
        Y = np.empty((max(M, 1), 4))
        Y[:, 0] = np.sort(rs.uniform(0.25, 1.75, max(M, 1)))
        Y[:, 1:] = c + 0.01 * rs.normal(size=(max(M, 1), 3))
        pair = []
        for withm in (True, False):
            X = np.vstack([bg, Y]) if (withm and M > 0) else bg
            R, U, G, CP, infl, h = prep(X, rs)
            rr = np.linalg.norm(X[:, 1:], axis=1)
            pool = np.flatnonzero((rr < SIG) & (rr > 0.03)
                                  & (U.sum(1) >= 3))
            if len(pool) < 25:
                pair.append(np.nan); del R, U, X; continue
            hv, _, _ = sample(X, U, G, CP, infl, h, pool, 300, 6, rs,
                              norm=True)
            del R, U, X
            pair.append(float(np.mean(hv)) if len(hv) else np.nan)
        if np.isfinite(pair[0]) and np.isfinite(pair[1]):
            dif.append((pair[0], pair[1]))
    if len(dif) < 3:
        print(f'{M:>11}  мало'); continue
    D = np.array(dif)
    d = D[:, 0] - D[:, 1]
    se = d.std(ddof=1) / math.sqrt(len(d))
    print(f'{M:>11}{D[:,0].mean():>16.4f}{D[:,1].mean():>12.4f}'
          f'{d.mean():>+10.4f}'
          f'{(abs(d.mean())/se if se > 1e-12 else float("nan")):>7.1f}')
