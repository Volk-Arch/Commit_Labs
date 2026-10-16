#!/usr/bin/env python3
"""ЧЁРНАЯ ДЫРА ИЗОГНУТЫМ СРЕЗОМ (3+1). Без лоскутов и швов.

Объемлющее пространство — след золотой решётки: R^{1,7}, одно время T, три оси x,y,z
и четыре теневые оси T', x', y', z' (все пространственные).

Срез Σ = внешность Шварцшильда, вложенная так (вариант Фронсдала с поворотом вместо буста):
    T = t,  (x, y, z) = r·n̂  (r — площадной радиус),
    x' + i y' = A(r)·e^{iωt},  A = √(2M/r)/ω,
    z' = B(r),  B'² = 2M/(r − 2M) − A'²
Индуцированная метрика: (1 − A²ω²)dt² − (1 + A'² + B'²)dr² − r²dΩ² — точно Шварцшильд.
Время дыры — винтовое движение: сдвиг по T и поворот в тени (x', y').

Правило:  событие есть ⟺ |смещение от Σ по нормали|² + T'² ≤ r_окна²  (шар, как в gold4);
          положение — проекция на Σ.
Проверка — правило вакуума как в vacuum.py: ромбы вдоль радиуса и поперёк, r = 10M.
Эйнштейн обязан дать нулевую анизотропию.
"""
import sys, math, time, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.linalg import expm
from gold4 import r_for_density
from vacuum import M as M0, RC, TAU, V, G, boost_matrix
from bentcut import zt_range, ST, SI

RHO0 = 30.0
OMEGA = float(os.environ.get('OMEGA', 0.25))

class Hole:
    def __init__(s, M, omega=OMEGA):
        s.M, s.w = M, omega
    def A(s, r): return np.sqrt(2 * s.M / r) / s.w if s.M > 0 else 0 * r
    def Ap(s, r): return -0.5 * np.sqrt(2 * s.M) * r ** -1.5 / s.w if s.M > 0 else 0 * r
    def Bp(s, r):
        if s.M == 0: return 0 * r
        v = 2 * s.M / (r - 2 * s.M) - s.Ap(r) ** 2
        return np.sqrt(np.maximum(v, 0))
    def B(s, r):
        # B(r) = ∫ B'(u) du, отсчёт от RC (константа роли не играет)
        if s.M == 0: return 0 * r
        if not hasattr(s, '_u'):
            u = np.linspace(2.5 * s.M, 60, 120001); bp = s.Bp(u)
            cum = np.concatenate([[0], np.cumsum(0.5 * (bp[1:] + bp[:-1]) * np.diff(u))])
            s._u, s._cum = u, cum - np.interp(RC, u, cum)
        return np.interp(r, s._u, s._cum)
    def surf(s, t, X):
        r = np.linalg.norm(X, axis=-1); A = s.A(r)
        return np.concatenate([t[..., None], X, (A * np.cos(s.w * t))[..., None], (A * np.sin(s.w * t))[..., None], s.B(r)[..., None]], -1)
    def tangents(s, t, X):
        """∂S/∂(t, X1, X2, X3): (n, 4, 7)"""
        r = np.linalg.norm(X, axis=-1); nh = X / r[..., None]
        A, Ap, Bp = s.A(r), s.Ap(r), s.Bp(r); c, sn = np.cos(s.w * t), np.sin(s.w * t)
        n = len(t); D = np.zeros((n, 4, 7))
        D[:, 0, 0] = 1; D[:, 0, 4] = -A * s.w * sn; D[:, 0, 5] = A * s.w * c
        for i in range(3):
            D[:, 1 + i, 1 + i] = 1
            D[:, 1 + i, 4] = Ap * nh[:, i] * c; D[:, 1 + i, 5] = Ap * nh[:, i] * sn; D[:, 1 + i, 6] = Bp * nh[:, i]
        return D

ETA7 = np.array([1, -1, -1, -1, -1, -1, -1.0])

def project(hole, P, iters=6):
    """ближайшая (по нормали) точка Σ: параметры (t, X) и квадрат нормального смещения."""
    t = P[:, 0].copy(); X = P[:, 1:4].copy(); bad = np.zeros(len(t), bool)
    for _ in range(iters):
        S = hole.surf(t, X); D = hole.tangents(t, X); d = P - S
        DE = D * ETA7
        g = DE @ D.transpose(0, 2, 1)
        b = ~(np.abs(np.linalg.det(g)) > 1e-8) | ~np.isfinite(g).all((1, 2))   # вырожденная рамка (r у 2M): выкинуть
        bad |= b; g[b] = np.diag([1.0, -1, -1, -1]); d[b] = 0; DE[b] = 0
        du = np.linalg.solve(g, DE @ d[..., None])[..., 0]
        t += du[:, 0]; X += du[:, 1:]
    d = P - hole.surf(t, X)
    n2 = -((d * d) @ ETA7); n2[bad | ~np.isfinite(n2)] = np.inf               # нормаль пространственная: |N|² ≥ 0
    return t, X, n2

def iso_from_areal(X, M):
    r = np.linalg.norm(X, axis=-1)
    rho = (r - M + np.sqrt(np.maximum(r * r - 2 * M * r, 0))) / 2
    return X * (rho / r)[:, None]

def areal_from_iso(X, M):
    rho = np.linalg.norm(X, axis=-1); r = rho * (1 + M / (2 * rho)) ** 2
    return X * (r / rho)[..., None]

def earlier_pts(a, X, M, sign=1):
    """первый порядок, γ = 1, изотропные координаты (как в vacuum.py)"""
    d = (X - a) * sign; D = np.linalg.norm(d[:, 1:], axis=1)
    ra = np.linalg.norm(a[1:]); rx = np.linalg.norm(X[:, 1:], axis=1)
    S = np.log((ra + rx + D) / np.maximum(ra + rx - D, 1e-12))
    return d[:, 0] > D + 2 * M * S

def proper(p, q, M, k=200):
    s = (np.arange(k) + 0.5) / k; pts = p[None] + s[:, None] * (q - p)[None]
    ph = -M / np.linalg.norm(pts[:, 1:], axis=1); dt = q[0] - p[0]; dx2 = ((q[1:] - p[1:]) ** 2).sum()
    return np.mean(np.sqrt((1 + 2 * ph) * dt * dt - (1 - 2 * ph) * dx2))

def lattice_box(box, sh_ranges, rw, hole_bounds):
    """точки решётки (система решётки): физика в box, тень — прогрессивный отбор по нижней оценке
    расстояния до теневой части Σ. Возвращает (Y_phys (n,4), Y_shadow (n,4))."""
    pass
    vals = []
    t, tp = zt_range(box[0][0] / ST, box[0][1] / ST, sh_ranges[0][0] / SI, sh_ranges[0][1] / SI); vals.append((t * ST, tp * SI))
    for k in range(1, 4):
        c, cp = zt_range(box[k][0], box[k][1], sh_ranges[k][0], sh_ranges[k][1]); vals.append((c, cp))
    return vals

def gen_points(hole, c_areal, Hbox, Lm, sh, rw, prefilter):
    """события Σ около c_areal. Lm — изометрия решётки (физика), sh — сдвиг (8)."""
    M = hole.M
    rr = np.linalg.norm(c_areal[1:]); rlo, rhi = max(rr - 1.2 * Hbox, 3 * M + 1e-9), rr + 1.2 * Hbox
    Amax = float(hole.A(np.array(rlo))) if M > 0 else 0.0
    Bmin, Bmax = (float(hole.B(np.array(rlo))), float(hole.B(np.array(rhi)))) if M > 0 else (0.0, 0.0)
    # физический ящик в системе решётки
    corners = np.array([[a, b, cc, d] for a in (-Hbox, Hbox) for b in (-Hbox, Hbox) for cc in (-Hbox, Hbox) for d in (-Hbox, Hbox)])
    Li = np.linalg.inv(Lm)
    Yc = corners @ Li.T
    lo, hi = Yc.min(0) + sh[:4], Yc.max(0) + sh[:4]
    shr = [(-rw, rw), (-Amax - rw, Amax + rw), (-Amax - rw, Amax + rw), (Bmin - rw, Bmax + rw)]
    shr = [(a + sh[4 + k], b + sh[4 + k]) for k, (a, b) in enumerate(shr)]
    (t, tq), (x, xq), (y, yq), (z, zq) = lattice_box([(lo[k], hi[k]) for k in range(4)], shr, rw, None)
    tq, xq, yq, zq = tq - sh[4], xq - sh[5], yq - sh[6], zq - sh[7]
    def dint(v, a, b): return np.maximum(0, np.maximum(a - v, v - b))
    r2 = rw * rw * 1.02
    # прогрессивное произведение: (t), x, y, z с нижней оценкой теневого расстояния
    W = tq ** 2; I = [np.arange(len(t))]
    keep = W <= r2; idx = np.nonzero(keep)[0]; W = W[idx]; cols = [idx]
    for (vv, vq, a, b) in ((x, xq, -Amax, Amax), (y, yq, -Amax, Amax), (z, zq, Bmin, Bmax)):
        dq = dint(vq, a, b) ** 2
        n0 = len(W)
        ii = np.repeat(np.arange(n0), len(vv)); jj = np.tile(np.arange(len(vv)), n0)
        Wn = W[ii] + dq[jj]; k = Wn <= r2
        ii, jj, W = ii[k], jj[k], Wn[k]
        cols = [c_[ii] for c_ in cols] + [jj]
    it, ix, iy, iz = cols
    Yp = np.stack([t[it], x[ix], y[iy], z[iz]], 1); Ys = np.stack([tq[it], xq[ix], yq[iy], zq[iz]], 1)
    Xp = (Yp - sh[:4]) @ Lm.T + c_areal                     # физика в объемлющем
    m = prefilter(Xp); Xp, Ys = Xp[m], Ys[m]
    P = np.concatenate([Xp, Ys[:, 1:]], 1)                   # 7 координат объемлющего (без T')
    tt, XX, n2 = project(hole, P)
    acc = n2 + Ys[:, 0] ** 2 <= rw * rw
    return np.concatenate([tt[acc, None], XX[acc]], 1)

def lattice_iso(rs):
    Gm = np.zeros((4, 4)); bv = rs.normal(size=3); bv *= 0.6 / np.linalg.norm(bv); Gm[0, 1:] = Gm[1:, 0] = bv
    R = np.zeros((4, 4)); r3 = rs.normal(0, 1.5, 3); R[1, 2], R[2, 1], R[1, 3], R[3, 1], R[2, 3], R[3, 2] = r3[0], -r3[0], r3[1], -r3[1], r3[2], -r3[2]
    return expm(Gm) @ expm(R)

def one(M, kind, rs, crystal=True):
    hole = Hole(M)
    nhat = rs.normal(size=3); nhat /= np.linalg.norm(nhat); c = np.concatenate([[0.0], RC * nhat])
    w = np.cross(nhat, rs.normal(size=3)); w /= np.linalg.norm(w)
    n = nhat if kind == 'вдоль r' else w
    B = boost_matrix(n, V); u = B[:, 0].copy(); ph_c = -M / RC
    u[0] /= math.sqrt(1 + 2 * ph_c); u[1:] /= math.sqrt(1 - 2 * ph_c)
    p = c - TAU * u / 2; q = c + TAU * u / 2; H = 1.06 * TAU    # c, p, q — изотропные координаты
    Bi = np.linalg.inv(B)
    c_ar = np.concatenate([[0.0], areal_from_iso(c[1:], M)]) if M > 0 else c.copy()
    hd = 0.55 * H * 1.02                                     # полувысота с запасом (ромб: τ/2 ≈ 0.5 H)
    sB = np.linalg.norm(Bi, 2)
    def prefilter(Xa, m):                                    # в рамке ромба: |T| + |x| ≤ hd (+ сдвиг m в объемлющем)
        Xi = Xa.copy()
        if M > 0: Xi[:, 1:] = iso_from_areal(Xa[:, 1:], M)
        Y = (Xi - c) @ Bi.T
        return np.abs(Y[:, 0]) + np.linalg.norm(Y[:, 1:], axis=1) <= hd + 2 * sB * m
    if crystal:
        Lm = B @ lattice_iso(rs)                                 # решётка: случайный буст 0.6 и поворот относительно ромба
        # полуразмер ящика: ромб в рамке B помещается в куб ±H/2; в системе решётки — через Lm
        sh = np.concatenate([rs.uniform(0, 400, 4), rs.uniform(-3, 3, 4)])
        rw = r_for_density(RHO0)
        Xa = gen_points_frame(hole, c_ar, hd, B, Lm, sh, rw, prefilter, disp=(0.75 if M > 0 else 0.05))
    else:
        Xa = poisson_points(hole, c, B, H, rs)
    Xi = Xa.copy()
    if M > 0: Xi[:, 1:] = iso_from_areal(Xa[:, 1:], M)
    inside = earlier_pts(p, Xi, M) & earlier_pts(q, Xi, M, sign=-1)
    N = inside.sum(); tau = proper(p, q, M)
    V0 = RHO0 * math.pi / 24 * tau ** 4
    return N / V0 - 1

def gen_points_frame(hole, c_ar, h, B, Lm, sh, rw, prefilter, disp=0.7, chunk=40):
    """события Σ в ромбе (полувысота h в рамке B) около c_ar. Ящик решётки — по самому ромбу,
    раздутому на disp (сдвиг точки при проекции на Σ ~ √(2M/r)·r_окна)."""
    M = hole.M
    dirs = np.random.default_rng(0).normal(size=(400, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    Vd = np.concatenate([np.array([[h, 0, 0, 0], [-h, 0, 0, 0]]), np.concatenate([np.zeros((400, 1)), h * dirs], 1)])
    Va = Vd @ B.T                                            # смещения вершин в объемлющем
    Li = np.linalg.inv(Lm); Yc = Va @ Li.T
    marg = disp * np.linalg.norm(Li, 2)
    lo, hi = Yc.min(0) - marg + sh[:4], Yc.max(0) + marg + sh[:4]
    rr = np.linalg.norm((Va + c_ar)[:, 1:], axis=1); rlo, rhi = rr.min() - disp - 0.5, rr.max() + disp + 0.5
    if M > 0: assert OMEGA ** 2 >= (rlo - 2 * M) / (4 * rlo ** 3), 'ω мала для этого rlo'
    Amax = float(hole.A(np.array(rlo))) if M > 0 else 0.0
    Bmin, Bmax = (float(hole.B(np.array(rlo))), float(hole.B(np.array(rhi)))) if M > 0 else (0.0, 0.0)
    shr = [(-rw, rw), (-Amax - rw, Amax + rw), (-Amax - rw, Amax + rw), (Bmin - rw, Bmax + rw)]
    shr = [(a + sh[4 + k], b + sh[4 + k]) for k, (a, b) in enumerate(shr)]
    (t, tq), (x, xq), (y, yq), (z, zq) = lattice_box([(lo[k], hi[k]) for k in range(4)], shr, rw, None)
    tq, xq, yq, zq = tq - sh[4], xq - sh[5], yq - sh[6], zq - sh[7]
    def dint(v, a, b): return np.maximum(0, np.maximum(a - v, v - b))
    r2 = rw * rw * 1.02
    dx, dy, dz = dint(xq, -Amax, Amax) ** 2, dint(yq, -Amax, Amax) ** 2, dint(zq, Bmin, Bmax) ** 2
    ok_t = np.nonzero(tq ** 2 <= r2)[0]
    outs = []
    for k0 in range(0, len(ok_t), chunk):
        it0 = ok_t[k0:k0 + chunk]; W = tq[it0] ** 2; cols = [it0]
        for dq, n in ((dx, len(x)), (dy, len(y)), (dz, len(z))):
            n0 = len(W); ii = np.repeat(np.arange(n0), n); jj = np.tile(np.arange(n), n0)
            Wn = W[ii] + dq[jj]; kk = Wn <= r2
            cols = [c_[ii[kk]] for c_ in cols] + [jj[kk]]; W = Wn[kk]
        it, ix, iy, iz = cols
        if len(it) == 0: continue
        Yp = np.stack([t[it], x[ix], y[iy], z[iz]], 1); Ys = np.stack([tq[it], xq[ix], yq[iy], zq[iz]], 1)
        Xp = (Yp - sh[:4]) @ Lm.T + c_ar
        # дешёвый отсев: точка дальше disp от ромба в объемлющем не попадёт
        m = prefilter(Xp, disp)
        if not m.any(): continue
        Xp, Ys = Xp[m], Ys[m]
        P = np.concatenate([Xp, Ys[:, 1:]], 1)
        t2, X2, n2 = project(hole, P, iters=2)              # две итерации -> точный отсев
        m = (n2 + Ys[:, 0] ** 2 <= 1.05 * rw * rw) & prefilter(np.concatenate([t2[:, None], X2], 1), 0.05)
        P, Ys = P[m], Ys[m]
        tt, XX, n2 = project(hole, P)
        acc = n2 + Ys[:, 0] ** 2 <= rw * rw
        o = np.concatenate([tt[acc, None], XX[acc]], 1)
        outs.append(o[prefilter(o, 0.0)])
    return np.concatenate(outs) if outs else np.zeros((0, 4))

def poisson_points(hole, c, B, H, rs):
    """засев с точной густотой Шварцшильда (√−g в изотропных координатах) — контроль."""
    M = hole.M
    sqmax = 1.1
    nexp = RHO0 * sqmax * (math.pi / 24) * H ** 4; n_pts = rs.poisson(nexp); Z = []
    while sum(len(z) for z in Z) < n_pts:
        Y = rs.uniform(-H / 2, H / 2, (8 * n_pts + 100, 4)); Y = Y[np.abs(Y[:, 0]) + np.linalg.norm(Y[:, 1:], axis=1) <= H / 2]; Z.append(Y)
    Y = np.concatenate(Z)[:n_pts]; X = Y @ B.T + c
    rho = np.linalg.norm(X[:, 1:], axis=1)
    sq = (1 - M / (2 * rho)) / (1 + M / (2 * rho)) * (1 + M / (2 * rho)) ** 6 if M > 0 else np.ones(len(X))
    X = X[rs.random(len(X)) < sq / sqmax]
    if M > 0: X[:, 1:] = areal_from_iso(X[:, 1:], M)
    return X

if __name__ == '__main__':
    NDIA = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    Mv = float(sys.argv[2]) if len(sys.argv) > 2 else M0
    what = sys.argv[3] if len(sys.argv) > 3 else 'crystal'
    kinds = sys.argv[4].split(',') if len(sys.argv) > 4 else ['вдоль r', 'поперёк']
    SOFF = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    t0 = time.time(); res = {}
    print(f'M = {Mv}, ω = {OMEGA}, r = {RC}, τ = {TAU}, v = {V}; {what}; ромбов {NDIA}'); sys.stdout.flush()
    for kind in kinds:
        rs = np.random.default_rng(777 + (2 if kind == 'вдоль r' else 3) + 1000 * SOFF + (0 if what == 'crystal' else 50000))
        d = []
        for i in range(NDIA):
            d.append(one(Mv, kind, rs, crystal=(what == 'crystal')))
            if (i + 1) % max(1, NDIA // 5) == 0:
                print(f'   {kind}: {i + 1} ромбов, δ = {np.mean(d):+.5f} ± {np.std(d, ddof=1) / math.sqrt(len(d)):.5f}  [{time.time() - t0:.0f} с]'); sys.stdout.flush()
        d = np.array(d); res[kind] = (d.mean(), d.std(ddof=1) / math.sqrt(NDIA))
        np.save(f'/home/claude/data/blackcut_{what}_{"along" if kind == "вдоль r" else "perp"}_M{Mv}_s{SOFF}.npy', d)
        print(f'срез ({what}) {kind:8s}: δ = {d.mean():+.5f} ± {res[kind][1]:.5f} (разброс на ромб {d.std(ddof=1):.4f})   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
    if len(res) == 2:
        a, b = res['вдоль r'], res['поперёк']
        print(f'   анизотропия {a[0] - b[0]:+.5f} ± {math.hypot(a[1], b[1]):.5f}  (Эйнштейн: 0)')
