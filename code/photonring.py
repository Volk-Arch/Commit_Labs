"""Фотонная сфера и тень из порядка. Экватор Шварцшильда (t, x, y), M = 1, события равномерны
(√−g = 1). Местное «раньше» по конусу Шварцшильда. Вспышка в точке на радиусе r_s; разрез по
положительной оси x запрещает рёбра через него. Первое прибытие к той же точке «с другой
стороны разреза» = самая быстрая петля света вокруг дыры через эту точку.
Теория: быстрее всего свет обходит дыру по фотонной сфере r = 3M за 2π·3√3·M = 32.65M,
а радиус тени b = 3√3 M = 5.196M — это та же петля, делённая на 2π."""
import sys, math, time, json, numpy as np
sys.path.insert(0, '.')
from chains import local_edges
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order
RHO = float(sys.argv[1]) if len(sys.argv) > 1 else 30; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
EPS = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2; T = 60.0; RLO, RHI = 2.05, 7.5
rng = np.random.default_rng(SEED); t0 = time.time()
def ok(mid, d):
    r = np.hypot(mid[:, 1], mid[:, 2]); A = 1 - 2 / r
    dr = (mid[:, 1] * d[:, 1] + mid[:, 2] * d[:, 2]) / r
    rdphi = (mid[:, 1] * d[:, 2] - mid[:, 2] * d[:, 1]) / r
    # разрез: отрезок не пересекает луч y = 0, x > 0
    a, b = mid - d / 2, mid + d / 2
    cross = (np.sign(a[:, 2]) != np.sign(b[:, 2])) & ((a[:, 1] + b[:, 1]) > 0)
    return (A > 0) & (A * d[:, 0] ** 2 > dr ** 2 / A + rdphi ** 2) & ~cross
n = rng.poisson(RHO * T * math.pi * (RHI ** 2 - RLO ** 2))
u = rng.random(n); rr = np.sqrt(RLO ** 2 + u * (RHI ** 2 - RLO ** 2)); ph = rng.uniform(-math.pi, math.pi, n)
P = np.column_stack([rng.uniform(0, T, n), rr * np.cos(ph), rr * np.sin(ph)])
RS = [2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 5.0, 6.0]
src = np.array([[0.0, r * math.cos(0.03), r * math.sin(0.03)] for r in RS])
P = np.vstack([src, P]); o = np.argsort(P[:, 0], kind='stable'); inv = np.empty_like(o); inv[o] = np.arange(len(o)); P = P[o]
ei, ej = local_edges(P, ok, EPS, slab=1.5)
G = csr_matrix((np.ones(len(ei), bool), (ei, ej)), shape=(len(P), len(P)))
r_all = np.hypot(P[:, 1], P[:, 2]); ph_all = np.arctan2(P[:, 2], P[:, 1])
print(f'событий {len(P)}, рёбер {len(ei)}   [{time.time()-t0:.0f} с]')
out = {}
for k, r in enumerate(RS):
    reach = breadth_first_order(G, int(inv[k]), directed=True, return_predecessors=False)
    print(f'   достижимо {len(reach)}, позднейшее t {P[reach,0].max():.1f}', end=' ')
    cell = reach[(ph_all[reach] < 0) & (ph_all[reach] > -0.15) & (np.abs(r_all[reach] - r) < 0.2)]
    tarr = float(P[cell, 0].min()) if len(cell) else float('nan')
    # за ту же петлю: время от вспышки до ближайшего края (без обхода) — для контроля
    out[r] = tarr
    print(f'вспышка на r = {r:.2f}M: самая быстрая петля вокруг дыры = {tarr:.2f}M   (по окружности этого радиуса: {2*math.pi*r/math.sqrt(1-2/r):.2f})'); sys.stdout.flush()
best = min(out, key=lambda k: out[k] if out[k] == out[k] else 1e9)
print(f'ИТОГ: быстрее всего петля через r = {best}M: {out[best]:.2f}M (теория: r = 3M, 32.65M); радиус тени b = петля/2π = {out[best]/2/math.pi:.3f}M (теория 5.196M)')
json.dump({str(k): v for k, v in out.items()}, open(f'/home/claude/data/photonring_rho{RHO:g}_s{SEED}.json', 'w'))
