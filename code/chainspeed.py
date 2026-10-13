"""Сколько звеньев в самой длинной цепочке на единицу собственного времени в зависимости от
скорости отрезка A→B относительно кристалла (2+1, плоско). Лоренц ⇒ не зависит от η."""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
from chains import local_edges, longest_from
from gold4 import coord_values, ST, SI
RHO = 20.0; r_win = (RHO * 5 * math.sqrt(5) * 3 / (4 * math.pi)) ** (1 / 3); TAU0 = 14.0
ok = lambda mid, d: d[:, 0] ** 2 > d[:, 1] ** 2 + d[:, 2] ** 2
def crystal_box(lo, hi, sh):
    t, tp = coord_values((lo[0] + sh[0]) / ST, (hi[0] + sh[0]) / ST, r_win / SI); vals = [(t * ST - sh[0], (tp * SI) ** 2)]
    for k in (1, 2):
        cc, cp = coord_values(lo[k] + sh[k], hi[k] + sh[k], r_win); vals.append((cc - sh[k], cp ** 2))
    P = np.zeros((1, 3)); W = np.zeros(1)
    for k, (cc, w2) in enumerate(vals):
        m = len(P); P = np.repeat(P, len(cc), 0); W = np.repeat(W, len(cc))
        P[:, k] = np.tile(cc, m); W += np.tile(w2, m); keep = W <= r_win ** 2; P, W = P[keep], W[keep]
    return P
if __name__ == '__main__':
    rng = np.random.default_rng(3); t0 = time.time()
    print(f'собственное время отрезка {TAU0}; в таблице длина самой длинной цепочки / τ')
    for kind in ('кристалл', 'посыпка'):
        row = []
        for eta in (0.0, 0.4, 0.8, 1.2, 1.6):
            ls = []
            for s in range(6):
                th = rng.uniform(0, 2 * math.pi); B = np.array([TAU0 * math.cosh(eta), TAU0 * math.sinh(eta) * math.cos(th), TAU0 * math.sinh(eta) * math.sin(th)])
                lo = np.minimum(0, B) - 1; hi = np.maximum(0, B) + 1
                # берём только ромб между A и B (остальное не влияет)
                if kind == 'кристалл': P = crystal_box(lo, hi, rng.uniform(0, 300, 3))
                else:
                    n = rng.poisson(RHO * np.prod(hi - lo)); P = lo + rng.random((n, 3)) * (hi - lo)
                d1 = P; d2 = B - P
                keep = (d1[:, 0] ** 2 > (d1[:, 1:] ** 2).sum(1)) & (d1[:, 0] > 0) & (d2[:, 0] ** 2 > (d2[:, 1:] ** 2).sum(1)) & (d2[:, 0] > 0)
                P = np.vstack([[0, 0, 0], P[keep], B]); P = P[np.argsort(P[:, 0], kind='stable')]
                ei, ej = local_edges(P, ok, 1.2, slab=3.0); dA, _ = longest_from(P, 0, ei, ej); ls.append(dA[-1] / TAU0)
            row.append(f'η={eta}: {np.mean(ls):.2f}±{np.std(ls)/math.sqrt(len(ls)):.2f}')
        print(f'{kind:9s} ' + '  '.join(row) + f'   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
