"""Петля света, правило с шагом в собственном времени статического наблюдателя:
местное «раньше» если √A·Δt < ε (A в середине) и разность внутри конуса Шварцшильда.
Так у дыры шаг одинаков в местных единицах, и свет из цепочек отстаёт так же, как в
плоском мире (калибровка flatlight: ×1.08 при ρ = 30, ε = 1.2)."""
import sys, math, time, json, numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order
RHO = float(sys.argv[1]) if len(sys.argv) > 1 else 30; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
EPS = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2; T = 50.0; RLO, RHI = 2.6, 7.5; SLAB = 1.0
rng = np.random.default_rng(SEED); t0 = time.time()
def Af(r): return 1 - 2 / r
n = rng.poisson(RHO * T * math.pi * (RHI ** 2 - RLO ** 2))
u = rng.random(n); rr = np.sqrt(RLO ** 2 + u * (RHI ** 2 - RLO ** 2)); ph = rng.uniform(-math.pi, math.pi, n)
P = np.column_stack([rng.uniform(0, T, n), rr * np.cos(ph), rr * np.sin(ph)])
RS = [2.75, 3.0, 3.25, 3.5, 4.0, 5.0]
src = np.array([[0.0, r * math.cos(0.03), r * math.sin(0.03)] for r in RS])
P = np.vstack([src, P]); o = np.argsort(P[:, 0], kind='stable'); inv = np.empty_like(o); inv[o] = np.arange(len(o)); P = P[o]
t = P[:, 0]; R = np.hypot(P[:, 1], P[:, 2]); sA = np.sqrt(Af(R))
I, J = [], []
tmax_step = EPS / math.sqrt(Af(RLO))
s0 = 0.0
while s0 <= T:
    a = np.searchsorted(t, s0); b = np.searchsorted(t, s0 + SLAB); c = np.searchsorted(t, s0 + SLAB + tmax_step)
    if b > a:
        idx = np.arange(a, c)
        Y = np.column_stack([sA[idx] * (t[idx] - s0), P[idx, 1], P[idx, 2]])   # местное время
        pr = cKDTree(Y).query_pairs(r=EPS * 1.6, output_type='ndarray')
        if len(pr):
            i, j = idx[pr[:, 0]], idx[pr[:, 1]]
            sw = t[i] > t[j]; i2 = np.where(sw, j, i); j2 = np.where(sw, i, j); k = i2 < b; i2, j2 = i2[k], j2[k]
            d = P[j2] - P[i2]; mid = 0.5 * (P[j2] + P[i2]); rm = np.hypot(mid[:, 1], mid[:, 2]); A = Af(rm)
            dr = (mid[:, 1] * d[:, 1] + mid[:, 2] * d[:, 2]) / rm; rdphi = (mid[:, 1] * d[:, 2] - mid[:, 2] * d[:, 1]) / rm
            aa, bb = mid - d / 2, mid + d / 2
            cross = (np.sign(aa[:, 2]) != np.sign(bb[:, 2])) & ((aa[:, 1] + bb[:, 1]) > 0)
            ok = (d[:, 0] > 0) & (np.sqrt(A) * d[:, 0] < EPS) & (A * d[:, 0] ** 2 > dr ** 2 / A + rdphi ** 2) & ~cross
            I.append(i2[ok]); J.append(j2[ok])
    s0 += SLAB
ei, ej = np.concatenate(I), np.concatenate(J)
G = csr_matrix((np.ones(len(ei), bool), (ei, ej)), shape=(len(P), len(P)))
ph_all = np.arctan2(P[:, 2], P[:, 1])
print(f'событий {len(P)}, рёбер {len(ei)} ({len(ei)/len(P):.1f} на событие)   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
out = {}
for k, r in enumerate(RS):
    reach = breadth_first_order(G, int(inv[k]), directed=True, return_predecessors=False)
    cell = reach[(ph_all[reach] < 0) & (ph_all[reach] > -0.15) & (np.abs(R[reach] - r) < 0.2)]
    out[r] = float(P[cell, 0].min()) if len(cell) else float('nan')
    print(f'r = {r}: петля {out[r]:.2f}M'); sys.stdout.flush()
json.dump({str(k): v for k, v in out.items()}, open(f'/home/claude/data/photonring2_rho{RHO:g}_s{SEED}.json', 'w'))
