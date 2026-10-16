#!/usr/bin/env python3
"""ОРБИТЫ НА ИЗОГНУТОМ СРЕЗЕ (2+1, экваториальная плоскость Шварцшильда, M = 1).

Объемлющее: след золотой решётки 2+1 = R^{1,5}: (T, x, y | T', x', y'), тень пространственная.
Срез Σ:  T = t, (x, y) = r(cos φ, sin φ),  x' + i y' = A(r) e^{iωt},  T' = B(r)
         A = √(2/r)/ω,  B'² = 2/(r − 2) − A'²   =>  на Σ точно A dt² − dr²/A − r² dφ².
Событие есть ⟺ его смещение от Σ по нормали (3 нормали) в шаре окна (как в плоском мире);
положение — проекция на Σ. Решётка повёрнута и бустнута (0.8) относительно дыры.
Проверка — та же, что в bend.py: трубка самых длинных цепочек, дуга 135°, R = 8M.
"""
import sys, math, time, json, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chains import local_edges, longest_from, longest_to
from bentcut import zt_range, ST, SI

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
OM = float(sys.argv[2]) if len(sys.argv) > 2 else 0.12
SEED0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
R = 8.0; RHO = 20.0; EPS = 1.0; DPHI = 0.75 * math.pi; FR = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
TK = DPHI * math.sqrt(R ** 3); TJ = TK * FR; TEND = TJ[-1] + 1
rlo, rhi = 0.55 * R, 1.45 * R; a0, a1 = -math.radians(25), DPHI + math.radians(25)
RW = (RHO * 5 * math.sqrt(5) * 3 / (4 * math.pi)) ** (1 / 3)
ETA6 = np.array([1, -1, -1, -1, -1, -1.0])

def Af(r): return 1 - 2 / r
def ok(mid, d):
    r = np.hypot(mid[:, 1], mid[:, 2]); A = Af(r)
    dr = (mid[:, 1] * d[:, 1] + mid[:, 2] * d[:, 2]) / r; rdphi = (mid[:, 1] * d[:, 2] - mid[:, 2] * d[:, 1]) / r
    return (A > 0) & (A * d[:, 0] ** 2 > dr ** 2 / A + rdphi ** 2)

class Hole2:
    def __init__(s, w):
        s.w = w
        u = np.linspace(2.2, 40, 200001); bp = s.Bp(u)
        cum = np.concatenate([[0], np.cumsum(0.5 * (bp[1:] + bp[:-1]) * np.diff(u))])
        s._u, s._cum = u, cum - np.interp(R, u, cum)
    def A(s, r): return np.sqrt(2 / r) / s.w
    def Ap(s, r): return -0.5 * math.sqrt(2) * r ** -1.5 / s.w
    def Bp(s, r): return np.sqrt(np.maximum(2 / (r - 2) - s.Ap(r) ** 2, 0))
    def B(s, r): return np.interp(r, s._u, s._cum)
    def surf(s, t, X):
        r = np.linalg.norm(X, axis=-1); A = s.A(r)
        return np.column_stack([t, X, A * np.cos(s.w * t), A * np.sin(s.w * t), s.B(r)])
    def tangents(s, t, X):
        r = np.linalg.norm(X, axis=-1); nh = X / r[:, None]
        A, Ap, Bp = s.A(r), s.Ap(r), s.Bp(r); c, sn = np.cos(s.w * t), np.sin(s.w * t)
        D = np.zeros((len(t), 3, 6))
        D[:, 0, 0] = 1; D[:, 0, 3] = -A * s.w * sn; D[:, 0, 4] = A * s.w * c
        for i in range(2):
            D[:, 1 + i, 1 + i] = 1
            D[:, 1 + i, 3] = Ap * nh[:, i] * c; D[:, 1 + i, 4] = Ap * nh[:, i] * sn; D[:, 1 + i, 5] = Bp * nh[:, i]
        return D

def project(h, P, iters=30):
    t = P[:, 0].copy(); X = P[:, 1:3].copy()
    def floor_r(X, rmin):
        r = np.linalg.norm(X, axis=1); k = r < rmin; X[k] *= (rmin / r[k])[:, None]; return X
    X = floor_r(X, 3.0)
    for _ in range(iters):
        D = h.tangents(t, X); d = P - h.surf(t, X); DE = D * ETA6
        g = DE @ D.transpose(0, 2, 1)
        du = np.linalg.solve(g, DE @ d[..., None])[..., 0]
        du = np.clip(du, -2, 2); t += du[:, 0]; X = floor_r(X + du[:, 1:], 2.3)
    d = P - h.surf(t, X); n2 = -((d * d) @ ETA6)
    res = np.abs(np.einsum('nik,nk->ni', h.tangents(t, X) * ETA6, d)).max(1)   # невязка ортогональности
    n2[(np.linalg.norm(X, axis=1) < 2.35) | (res > 1e-6)] = np.inf
    return t, X, n2

def build(rng, h):
    eta = 0.8; gm, gv = math.cosh(eta), math.sinh(eta); th = rng.uniform(0, 2 * math.pi)
    Rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    sh = np.concatenate([rng.uniform(0, 300, 3), rng.uniform(-3, 3, 3)])
    # прообраз области в системе решётки: (T, x, y) -> t_a = γT + γv x, u = γx + γv T, w = y; (X, Y) = Rot (u, w)
    corners = np.array([[ta, X, Y] for ta in (-2, TEND + 2) for X in (-rhi - 3, rhi + 3) for Y in (-rhi - 3, rhi + 3)])
    uw = corners[:, 1:] @ Rot                                  # Rot^T (X, Y)
    Bi = np.array([[gm, -gv], [-gv, gm]])                      # обратный буст в (t, u)
    tx = np.column_stack([corners[:, 0], uw[:, 0]]) @ Bi.T
    lo = np.array([tx[:, 0].min(), tx[:, 1].min(), uw[:, 1].min()]) + sh[:3]
    hi = np.array([tx[:, 0].max(), tx[:, 1].max(), uw[:, 1].max()]) + sh[:3]
    rl, rh = 1.0, rhi + 3                                       # ambient r (точка уходит от Σ на ~2.5 у дыры)
    Amin, Amax = float(h.A(np.array(rhi + 0.5))), float(h.A(np.array(rlo - 0.5)))
    Bmin, Bmax = float(h.B(np.array(rlo - 0.5))), float(h.B(np.array(rhi + 0.5)))
    t, tq = zt_range(lo[0] / ST, hi[0] / ST, (Bmin - RW + sh[3]) / SI, (Bmax + RW + sh[3]) / SI); T, Tq = t * ST - sh[0], tq * SI - sh[3]
    x, xq = zt_range(lo[1], hi[1], -Amax - RW + sh[4], Amax + RW + sh[4]); x, xq = x - sh[1], xq - sh[4]
    y, yq = zt_range(lo[2], hi[2], -Amax - RW + sh[5], Amax + RW + sh[5]); y, yq = y - sh[2], yq - sh[5]
    def dint(v, a, b): return np.maximum(0, np.maximum(a - v, v - b))
    dT = dint(Tq, Bmin, Bmax) ** 2
    r2 = RW * RW * 1.05; out = []
    for k0 in range(0, len(T), 8):
        it = np.arange(k0, min(k0 + 8, len(T)))
        ii = np.repeat(it, len(x)); jx = np.tile(np.arange(len(x)), len(it))
        ta = gm * T[ii] + gv * x[jx]
        m = (ta > -2) & (ta < TEND + 2) & (dT[ii] <= r2); ii, jx, ta = ii[m], jx[m], ta[m]
        # направление цели в тени зависит от времени ПРОЕКЦИИ, а оно отличается от ta на ≲ 3
        sp = OM * (ta[:, None] + np.linspace(-3.2, 3.2, 9)[None, :])
        c = np.cos(sp); cx_lo = np.minimum(Amin * c, Amax * c).min(1) - 0.1; cx_hi = np.maximum(Amin * c, Amax * c).max(1) + 0.1
        W = dT[ii] + dint(xq[jx], cx_lo, cx_hi) ** 2; m = W <= r2; ii, jx, ta, W = ii[m], jx[m], ta[m], W[m]
        if len(ii) == 0: continue
        sp = OM * (ta[:, None] + np.linspace(-3.2, 3.2, 9)[None, :])
        s_ = np.sin(sp); sy_lo = np.minimum(Amin * s_, Amax * s_).min(1) - 0.1; sy_hi = np.maximum(Amin * s_, Amax * s_).max(1) + 0.1
        n0 = len(ii); pa = np.repeat(np.arange(n0), len(y)); jy = np.tile(np.arange(len(y)), n0)
        W2 = W[pa] + dint(yq[jy], sy_lo[pa], sy_hi[pa]) ** 2; m = W2 <= r2; pa, jy = pa[m], jy[m]
        iT, iX = ii[pa], jx[pa]
        u = gm * x[iX] + gv * T[iT]; XY = np.column_stack([u, y[jy]]) @ Rot.T
        P = np.column_stack([ta[pa], XY, xq[iX], yq[jy], Tq[iT]])
        rr = np.hypot(P[:, 1], P[:, 2]); m = (rr > rl) & (rr < rh); P = P[m]
        if len(P) == 0: continue
        tt, XX, n2 = project(h, P)
        acc = n2 <= RW * RW
        Q = np.column_stack([tt[acc], XX[acc]])
        r = np.hypot(Q[:, 1], Q[:, 2]); ph = np.arctan2(Q[:, 2], Q[:, 1])
        out.append(Q[(Q[:, 0] >= 0) & (Q[:, 0] <= TEND) & (r > rlo) & (r < rhi) & (ph > a0) & (ph < a1)])
    return np.vstack(out)

if __name__ == '__main__':
    h = Hole2(OM); t0c = time.time(); allm = []; dens = []
    rb = np.linspace(rlo, rhi, 7)
    print(f'изогнутый срез 2+1: ω = {OM}, окно {RW:.3f}, плотность {RHO}'); sys.stdout.flush()
    for s in range(NS):
        rng = np.random.default_rng(900 + s + 100 * SEED0); P = build(rng, h); nw = len(P)
        rP = np.hypot(P[:, 1], P[:, 2])
        hist = np.histogram(rP, rb)[0] / (TEND * 0.5 * (a1 - a0) * np.diff(rb ** 2)) / RHO; dens.append(hist)
        special = np.array([[0, R, 0]] + [[tj, R * math.cos(DPHI), R * math.sin(DPHI)] for tj in TJ])
        P = np.vstack([special, P]); o = np.argsort(P[:, 0], kind='stable'); inv = np.empty_like(o); inv[o] = np.arange(len(o)); P = P[o]
        ei, ej = local_edges(P, ok, EPS); dA, _ = longest_from(P, int(inv[0]), ei, ej)
        rP = np.hypot(P[:, 1], P[:, 2]); phP = np.arctan2(P[:, 2], P[:, 1]); mids = []
        for k in range(len(TJ)):
            b = int(inv[1 + k]); dB = longest_to(P, b, ei, ej)
            tube = (dA >= 0) & (dB >= 0) & (dA + dB == dA[b]); w = tube & (np.abs(phP - DPHI / 2) < 0.15)
            mids.append(float(rP[w].mean()) if w.any() else float('nan'))
        allm.append(mids)
        print(f'мир {s}: событий {nw}, густота по r / ρ {np.round(hist, 3).tolist()}, радиус трубки {np.round(mids, 2)}   [{time.time() - t0c:.0f} с]'); sys.stdout.flush()
    A = np.array(allm)
    print(f'ИТОГ изогнутый срез ω={OM}: {np.round(np.nanmean(A, 0), 2).tolist()} ± {np.round(np.nanstd(A, 0) / math.sqrt(len(A)), 2).tolist()}  (точно 6.76, 7.38, 8.00, 8.62, 9.23)')
    print(f'густота по r (среднее): {np.round(np.mean(dens, 0), 4).tolist()}')
    json.dump(dict(omega=OM, mid=np.nanmean(A, 0).tolist(), err=(np.nanstd(A, 0) / math.sqrt(len(A))).tolist(), all=A.tolist(), dens=np.mean(dens, 0).tolist()),
              open(f'/home/claude/data/bentorbit_w{OM:g}_s{SEED0}.json', 'w'))
