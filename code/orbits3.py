"""Орбиты: золотой кристалл 2+1 против посыпки. Наблюдаемая — «трубка» всех самых длинных
цепочек из A в B (события e с dA(e) + dB(e) = L): радиус трубки в середине дуги 135°.
Кристалл: t, x, y ∈ Z[τ], тень (t', x', y') в шаре 0.618t'² + x'² + y'² ≤ r²; T = √τ·t.
Мир каждого «зерна» — свой сдвиг и поворот кристалла в плоскости xy."""
import sys, math, time, json, numpy as np
sys.path.insert(0, '.')
from chains import local_edges, longest_from, longest_to
from gold4 import coord_values, ST, SI
KIND = sys.argv[1] if len(sys.argv) > 2 else 'crystal'; NS = int(sys.argv[2]) if len(sys.argv) > 2 else 0; R = 8.0; RHO = 20.0; EPS = 1.0
DPHI = 0.75 * math.pi; FR = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
TK = DPHI * math.sqrt(R ** 3); TJ = TK * FR; T = TJ[-1] + 1
rlo, rhi = 0.55 * R, 1.45 * R; a0, a1 = -math.radians(25), DPHI + math.radians(25)
r_win = (RHO * 5 * math.sqrt(5) * 3 / (4 * math.pi)) ** (1 / 3)
def ok(mid, d):
    r = np.hypot(mid[:, 1], mid[:, 2]); A = 1 - 2 / r
    dr = (mid[:, 1] * d[:, 1] + mid[:, 2] * d[:, 2]) / r; rdphi = (mid[:, 1] * d[:, 2] - mid[:, 2] * d[:, 1]) / r
    return (A > 0) & (A * d[:, 0] ** 2 > dr ** 2 / A + rdphi ** 2)
def in_region(P):
    r = np.hypot(P[:, 1], P[:, 2]); ph = np.arctan2(P[:, 2], P[:, 1])
    return (r > rlo) & (r < rhi) & (ph > a0) & (ph < a1) & (P[:, 0] >= 0) & (P[:, 0] <= T)
def crystal(rng):
    """кристалл в своей системе, затем буст (быстрота 0.8, случайное направление) и поворот:
    общая система относительно дыры (в системе покоя кристалла цепочки идут по его столбцам)"""
    eta = 0.8; th = rng.uniform(0, 2 * math.pi); n = np.array([math.cos(th), math.sin(th)])
    B = np.eye(3); ch, sh_ = math.cosh(eta), math.sinh(eta)
    B[0, 0] = ch; B[0, 1:] = sh_ * n; B[1:, 0] = sh_ * n; B[1:, 1:] = np.eye(2) + (ch - 1) * np.outer(n, n)
    Bi = np.linalg.inv(B)
    corners = np.array([[tt, xx, yy] for tt in (0, T) for xx in (-rhi, rhi) for yy in (-rhi, rhi)]) @ Bi.T
    lo, hi = corners.min(0) - 0.5, corners.max(0) + 0.5; sh = rng.uniform(0, 300, 3)
    t, tp = coord_values((lo[0] + sh[0]) / ST, (hi[0] + sh[0]) / ST, r_win / SI)
    vals = [(t * ST - sh[0], (tp * SI) ** 2)]
    for k in (1, 2):
        cc, cp = coord_values(lo[k] + sh[k], hi[k] + sh[k], r_win); vals.append((cc - sh[k], cp ** 2))
    P = np.zeros((1, 3)); W = np.zeros(1)
    for k, (cc, w2) in enumerate(vals):
        m = len(P); P = np.repeat(P, len(cc), 0); W = np.repeat(W, len(cc))
        P[:, k] = np.tile(cc, m); W += np.tile(w2, m); keep = W <= r_win ** 2; P, W = P[keep], W[keep]
        if k == 0: continue
    P = P @ B.T
    return P[in_region(P)]
def poisson(rng):
    area = 0.5 * (a1 - a0) * (rhi ** 2 - rlo ** 2); n = rng.poisson(RHO * T * area)
    u = rng.random(n); rr = np.sqrt(rlo ** 2 + u * (rhi ** 2 - rlo ** 2)); ph = rng.uniform(a0, a1, n)
    return np.column_stack([rng.uniform(0, T, n), rr * np.cos(ph), rr * np.sin(ph)])
if __name__ == '__main__':
    t0 = time.time(); allm, allw = [], []
    for s in range(NS):
        rng = np.random.default_rng(500 + s)
        P = crystal(rng) if KIND == 'crystal' else poisson(rng)
        nworld = len(P)
        special = np.array([[0, R, 0]] + [[tj, R * math.cos(DPHI), R * math.sin(DPHI)] for tj in TJ])
        P = np.vstack([special, P]); o = np.argsort(P[:, 0], kind='stable'); inv = np.empty_like(o); inv[o] = np.arange(len(o)); P = P[o]
        ei, ej = local_edges(P, ok, EPS); dA, _ = longest_from(P, int(inv[0]), ei, ej)
        rP = np.hypot(P[:, 1], P[:, 2]); phP = np.arctan2(P[:, 2], P[:, 1])
        mids, widths = [], []
        for k in range(len(TJ)):
            b = int(inv[1 + k]); dB = longest_to(P, b, ei, ej)
            tube = (dA >= 0) & (dB >= 0) & (dA + dB == dA[b])
            w = tube & (np.abs(phP - DPHI / 2) < 0.15)
            mids.append(float(rP[w].mean()) if w.any() else float('nan')); widths.append(float(rP[w].std()) if w.sum() > 1 else 0.0)
        allm.append(mids); allw.append(widths)
        print(f'{KIND} зерно {s}: событий {nworld} (плотность {nworld/(T*0.5*(a1-a0)*(rhi**2-rlo**2)):.1f}), рёбер {len(ei)}, радиус трубки {np.round(mids,2)}, ширина {np.round(widths,2)}   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
    A = np.array(allm); m = np.nanmean(A, 0); sd = np.nanstd(A, 0)
    print(f'ИТОГ {KIND}: радиус трубки {np.round(m,2).tolist()}; разброс между мирами {np.round(sd,2).tolist()}; ширина трубки {np.round(np.mean(allw,0),2).tolist()}')
    json.dump(dict(kind=KIND, FR=FR.tolist(), mid=m.tolist(), sd=sd.tolist(), width=np.mean(allw, 0).tolist()), open(f'/home/claude/data/orbits3_{KIND}.json', 'w'))
