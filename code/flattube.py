"""Плоский 2+1: трубка самых длинных цепочек из A = 0 в B = (40, D, 0). Идёт ли она по прямой?
Кристалл (общая система) против посыпки, по 10 миров."""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
from chains import local_edges, longest_from, longest_to
sys.argv = sys.argv[:1]
import orbits3 as O
T, W = 30.0, 6.0; RHO = 20.0
O.T = T + 1; O.rhi = 22.0
def region(P): return (P[:, 0] >= 0) & (P[:, 0] <= T + 1) & (np.abs(P[:, 2]) < W) & (P[:, 1] > -W) & (P[:, 1] < 18)
O.in_region = region
ok = lambda mid, d: d[:, 0] ** 2 > d[:, 1] ** 2 + d[:, 2] ** 2
t0 = time.time()
for kind in ('crystal', 'poisson'):
    for D in (0.0, 12.0):
        offs = []
        for s in range(6):
            rng = np.random.default_rng(900 + s)
            if kind == 'crystal': P = O.crystal(rng)
            else:
                n = rng.poisson(RHO * (T + 1) * 2 * W * (18 + W)); P = np.column_stack([rng.uniform(0, T + 1, n), rng.uniform(-W, 18, n), rng.uniform(-W, W, n)])
            P = np.vstack([[0, 0, 0], [T, D, 0], P]); o = np.argsort(P[:, 0], kind='stable'); inv = np.empty_like(o); inv[o] = np.arange(len(o)); P = P[o]
            ei, ej = local_edges(P, ok, 1.0); dA, _ = longest_from(P, int(inv[0]), ei, ej); b = int(inv[1]); dB = longest_to(P, b, ei, ej)
            tube = (dA >= 0) & (dB >= 0) & (dA + dB == dA[b]); mid = tube & (np.abs(P[:, 0] - T / 2) < 1.0)
            # отклонение от прямой A→B в середине (по x и по y)
            off = P[mid, 1:].mean(0) - np.array([D / 2, 0]); offs.append(off)
        offs = np.array(offs)
        print(f'{kind:8s} D={D:4.0f}: среднее отклонение середины трубки от прямой ({offs[:,0].mean():+.2f}, {offs[:,1].mean():+.2f}), разброс {offs.std(0).round(2)}   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
