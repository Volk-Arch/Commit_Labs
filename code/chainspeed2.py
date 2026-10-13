"""То же без местного ограничения шага: полный порядок в ромбе A–B, самая длинная цепочка.
Буст вдоль оси x кристалла (там точная симметрия на η₀ = 1.0613) и в случайном направлении."""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
from chainspeed import crystal_box, RHO
TAU0 = 11.0; ETA0 = math.acosh((1 + 5 ** .5) / 2)
def longest_full(P):
    P = P[np.argsort(P[:, 0], kind='stable')]; n = len(P); dist = np.zeros(n, np.int32)
    for j in range(1, n):
        d = P[j] - P[:j]; pre = (d[:, 0] > 0) & (d[:, 0] ** 2 > (d[:, 1:] ** 2).sum(1))
        if pre.any(): dist[j] = dist[:j][pre].max() + 1
    return dist[-1]
if __name__ == '__main__':
    rng = np.random.default_rng(5); t0 = time.time()
    for kind in ('кристалл', 'посыпка'):
        for dirn in ('ось x', 'случайно'):
            if kind == 'посыпка' and dirn == 'ось x': continue
            row = []
            for eta in (0.0, 0.53, ETA0, 1.6, 2 * ETA0):
                ls = []
                for s in range(5):
                    th = 0.0 if dirn == 'ось x' else rng.uniform(0, 2 * math.pi)
                    B = np.array([TAU0 * math.cosh(eta), TAU0 * math.sinh(eta) * math.cos(th), TAU0 * math.sinh(eta) * math.sin(th)])
                    m = np.array([0.5, TAU0 / 2 + 0.5, TAU0 / 2 + 0.5]); lo = np.minimum(0, B) - m; hi = np.maximum(0, B) + m
                    if kind == 'кристалл': P = crystal_box(lo, hi, rng.uniform(0, 300, 3))
                    else:
                        n = rng.poisson(RHO * np.prod(hi - lo)); P = lo + rng.random((n, 3)) * (hi - lo)
                    d1 = P; d2 = B - P
                    keep = (d1[:, 0] > 0) & (d1[:, 0] ** 2 > (d1[:, 1:] ** 2).sum(1)) & (d2[:, 0] > 0) & (d2[:, 0] ** 2 > (d2[:, 1:] ** 2).sum(1))
                    ls.append((longest_full(np.vstack([[0, 0, 0], P[keep], B]))) / TAU0)
                row.append(f'{eta:.2f}: {np.mean(ls):.3f}±{np.std(ls)/math.sqrt(len(ls)):.3f}')
            print(f'{kind:9s} {dirn:8s} ' + '  '.join(row) + f'   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
