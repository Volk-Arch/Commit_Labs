"""Орбиты из порядка (вариант 2): дуга 135°, наблюдаемая — радиус цепочки в середине дуги.
A = (0, R, 0), B = (f·T_K, R·(cos135°, sin135°)), T_K = (3π/4)·√(R³/M). Сравнение с точной
геодезической. Вывод: T*, при котором цепочка идёт по окружности, и закон Кеплера T* ∝ R^1.5."""
import sys, math, time, json, numpy as np
sys.path.insert(0, '.')
from chains import local_edges, longest_from, path
R = float(sys.argv[1]); NS = int(sys.argv[2]); RHO = float(sys.argv[3]) if len(sys.argv) > 3 else 20
EPS = 1.0; DPHI = 0.75 * math.pi; FR = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
def ok(mid, d):
    r = np.hypot(mid[:, 1], mid[:, 2]); A = 1 - 2 / r
    dr = (mid[:, 1] * d[:, 1] + mid[:, 2] * d[:, 2]) / r
    rdphi = (mid[:, 1] * d[:, 2] - mid[:, 2] * d[:, 1]) / r
    return (A > 0) & (A * d[:, 0] ** 2 > dr ** 2 / A + rdphi ** 2)
TK = DPHI * math.sqrt(R ** 3); TJ = TK * FR; T = TJ[-1] + 1
rlo, rhi = 0.55 * R, 1.45 * R; a0, a1 = -math.radians(25), DPHI + math.radians(25)
area = 0.5 * (a1 - a0) * (rhi ** 2 - rlo ** 2)
t0 = time.time(); allmid = []; keep_path = None
for s in range(NS):
    rng = np.random.default_rng(1000 + s)
    n = rng.poisson(RHO * T * area)
    u = rng.random(n); rr = np.sqrt(rlo ** 2 + u * (rhi ** 2 - rlo ** 2)); ph = rng.uniform(a0, a1, n)
    P = np.column_stack([rng.uniform(0, T, n), rr * np.cos(ph), rr * np.sin(ph)])
    special = np.array([[0, R, 0]] + [[tj, R * math.cos(DPHI), R * math.sin(DPHI)] for tj in TJ])
    P = np.vstack([special, P]); o = np.argsort(P[:, 0], kind='stable'); inv = np.empty_like(o); inv[o] = np.arange(len(o)); P = P[o]
    ei, ej = local_edges(P, ok, EPS); dist, pred = longest_from(P, int(inv[0]), ei, ej)
    mids = []
    for k, d in enumerate(inv[1:len(special)]):
        Q = P[path(pred, d)]; phq = np.unwrap(np.arctan2(Q[:, 2], Q[:, 1])); rq = np.hypot(Q[:, 1], Q[:, 2])
        w = np.abs(phq - DPHI / 2) < 0.15                      # окно ±8.6° у середины
        mids.append(float(rq[w].mean()) if w.any() else float('nan'))
        if s == 0: (keep_path := keep_path or {}).__setitem__(f'{FR[k]}', Q[:, 1:].round(3).tolist())
    allmid.append(mids)
    print(f'R={R:.0f} зерно {s}: N={len(P)}, радиус в середине {np.round(mids, 2)}   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
A = np.array(allmid); m = np.nanmean(A, 0); e = np.nanstd(A, 0) / math.sqrt(len(A))
print(f'ИТОГ R={R:.0f}: T/T_K = {FR.tolist()}; радиус в середине = {np.round(m, 2).tolist()} ± {np.round(e, 2).tolist()}')
json.dump(dict(R=R, FR=FR.tolist(), mid=m.tolist(), err=e.tolist(), paths=keep_path), open(f'/home/claude/data/orbits2_R{R:g}.json', 'w'))
