"""Двигается ли порог насыщения вместе с масштабом сглаживания eps.

Если носитель веса по n равен ~1/eps, то набор eps читает разные
диапазоны плотности, и масса определяется по СПЕКТРУ, а не по числу.
"""
import math
import numpy as np
from weight import cone_rel, between, per_event
from weightmass import box_sprinkle, tube, T_MAX, RHO
from dent import n_for, boost, NEAR, FAR

L, RAD, S = 1.4, 0.05, 16
N = n_for(L)
EPSS = (0.05, 0.1, 0.2, 0.4)
MASSES = (12, 25, 50, 100, 200, 400, 800)


def depths(M, localized, seeds):
    out = {e: [] for e in EPSS}
    for t in range(seeds):
        rs = np.random.default_rng(500 + 17 * t)
        bg = box_sprinkle(N, L, rs)
        ob = tube(M, RAD, rs) if localized else box_sprinkle(M, L, rs)
        X = np.vstack([ob, bg])
        R = cone_rel(X); btw = between(R)
        Y = X[M:]
        keep = (np.abs(Y[:, 0] - T_MAX/2) < 1.2) & (np.abs(Y[:, 1]) < L)
        r = np.linalg.norm(Y[:, 1:], axis=1)
        near, far = keep & (r < NEAR), keep & (r >= FAR[0]) & (r < FAR[1])
        for e in EPSS:
            s, _ = per_event(R, btw, e)
            sv = s[M:]
            out[e].append(float(sv[near].mean() - sv[far].mean()))
        del R, btw, X
    return {e: (float(np.mean(v)), float(np.std(v)/math.sqrt(len(v))))
            for e, v in out.items()}


print(f'L={L}, N={N}, радиус {RAD}, засевов {S}')
print('глубина вмятины (трубка минус рассыпанный контроль той же массы)')
print()
print(f"{'M':>6}" + ''.join(f'{f"eps={e}":>20}' for e in EPSS))
tab = {}
for M in MASSES:
    dt, dc = depths(M, True, S), depths(M, False, S)
    cells = []
    for e in EPSS:
        d = dt[e][0] - dc[e][0]
        er = math.hypot(dt[e][1], dc[e][1])
        tab[(M, e)] = (d, er)
        cells.append(f'{d:>12.4f}/{abs(d)/max(er,1e-12):<7.1f}')
    print(f'{M:>6}' + ''.join(cells))
print('  (значение / значимость в сигмах)')
print()
print('ПОРОГ: наименьшая M, где значимость >= 3')
for e in EPSS:
    hit = [M for M in MASSES if abs(tab[(M, e)][0])/max(tab[(M, e)][1], 1e-12) >= 3]
    print(f'  eps={e}: ' + (f'M* = {hit[0]}' if hit else 'не достигнут'))
