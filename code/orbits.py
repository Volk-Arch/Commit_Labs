"""Орбиты из порядка. Экваториальная плоскость Шварцшильда (t, x, y), r = √(x²+y²), M = 1.
Объём √−g = 1 в (t, x, y) ⇒ события сеются равномерно. Местное «раньше»:
0 < Δt < ε и A·Δt² > Δr²/A + (r·Δφ)²,  A = 1 − 2M/r (середина отрезка).
Свободное падение = самая длинная цепочка. A = (0, R, φ=0), B_j = (T_j, R, φ=90°).
Если T_j равно четверти кеплеровского периода T_K = (π/2)·√(R³/M), цепочка должна идти по
окружности r = R; если короче — нырять внутрь, длиннее — выходить наружу.
Находим T*, при котором средний радиус цепочки = R, и проверяем T* ∝ R^1.5 (Кеплер)."""
import sys, math, time, json, numpy as np
sys.path.insert(0, '.')
from chains import local_edges, longest_from, path
t0 = time.time(); rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
RHO = float(sys.argv[1]) if len(sys.argv) > 1 else 10
EPS = 1.0
def ok(mid, d):
    r = np.hypot(mid[:, 1], mid[:, 2]); A = 1 - 2 / r
    dr = (mid[:, 1] * d[:, 1] + mid[:, 2] * d[:, 2]) / r
    rdphi = (mid[:, 1] * d[:, 2] - mid[:, 2] * d[:, 1]) / r
    return (A > 0) & (A * d[:, 0] ** 2 > dr ** 2 / A + rdphi ** 2)
res = {}
for R in (8.0, 12.0, 16.0):
    TK = (math.pi / 2) * math.sqrt(R ** 3)
    fr = np.array([0.8, 0.9, 1.0, 1.1, 1.2]); TJ = TK * fr
    T = TJ[-1] + 1
    # кольцо r ∈ [0.55R, 1.45R], угол от −25° до 115°
    rlo, rhi = 0.55 * R, 1.45 * R; a0, a1 = math.radians(-25), math.radians(115)
    area = 0.5 * (a1 - a0) * (rhi ** 2 - rlo ** 2)
    n = rng.poisson(RHO * T * area)
    u = rng.random(n); rr = np.sqrt(rlo ** 2 + u * (rhi ** 2 - rlo ** 2)); ph = rng.uniform(a0, a1, n)
    P = np.column_stack([rng.uniform(0, T, n), rr * np.cos(ph), rr * np.sin(ph)])
    special = np.array([[0, R, 0]] + [[tj, 0, R] for tj in TJ])
    P = np.vstack([special, P]); idx0 = np.arange(len(special))
    o = np.argsort(P[:, 0], kind='stable'); inv = np.empty_like(o); inv[o] = np.arange(len(o)); P = P[o]
    src = inv[0]; dsts = inv[1:]
    ei, ej = local_edges(P, ok, EPS)
    dist, pred = longest_from(P, src, ei, ej)
    rows = []
    for tj, f, d in zip(TJ, fr, dsts):
        if dist[d] < 0: rows.append((f, None, None)); continue
        Q = P[path(pred, d)]; rq = np.hypot(Q[:, 1], Q[:, 2])
        rows.append((f, float(rq.mean()), int(dist[d])))
        if abs(f - 1.0) < 1e-9 or abs(f - 0.8) < 1e-9 or abs(f - 1.2) < 1e-9:
            res.setdefault('paths', {})[f'{R}_{f}'] = Q[:, 1:].round(3).tolist()
    res[str(R)] = rows
    print(f'R = {R:.0f}M: T_K/4-периода = {TK:.1f}, событий {len(P)}, рёбер {len(ei)}; средний радиус цепочки при T/T_K = '
          + ', '.join(f'{f:.1f}: {m:.2f}' if m else f'{f:.1f}: —' for f, m, _ in rows) + f'   [{time.time()-t0:.0f} с]')
    sys.stdout.flush()
json.dump(res, open(f'/home/claude/data/orbits_rho{RHO:g}_s{sys.argv[2] if len(sys.argv)>2 else 1}.json', 'w'))
