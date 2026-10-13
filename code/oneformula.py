#!/usr/bin/env python3
"""ОДНА ФОРМУЛА: «в пустоте каждый малый ромб, в любой ориентации, содержит
столько событий, сколько плоский той же собственной высоты».
Проверяем, что она одна задаёт И закон спадания поля (Ньютон: 1/r), И двойку
(Эйнштейн: Ψ = Φ). Семейство: Φ = −A·(r₀/r)^p, Ψ = γΦ, Φ(r₀) = −0.03, r₀ = 10.
  δ_покой   — ромбы в покое: чувствуют ∇²Φ (ноль только при p = 1, поле Лапласа);
  δ_вдоль − δ_поперёк — анизотропия: чувствует Ψ − Φ (ноль только при γ = 1).
Мир строится обеими строками: густота ∝ √−g = 1 + Φ − 3Ψ; «раньше» — время света
∫(1 + (1+γ)|Φ|)dl по прямой (узлы Гаусса)."""
import sys, math, time
import numpy as np
import os
RC, TAU, V, RHO0 = float(os.environ.get('RC', 10)), float(os.environ.get('TAU', 6)), 0.8, float(os.environ.get('RHO0', 30))
AMP = float(os.environ.get('AMP', 0.03))
NDIA = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
G = 1 / math.sqrt(1 - V * V)
GL_x, GL_w = np.polynomial.legendre.leggauss(8); GL_x = 0.5 * (GL_x + 1); GL_w = 0.5 * GL_w
def make(p, gam):
    A = AMP
    def Phi(X): return -A * (RC / np.maximum(np.linalg.norm(X, axis=-1), 1e-6)) ** p
    def light_excess(a, X):
        """(1+γ)·∫|Φ| dl вдоль прямой a→x для всех X"""
        seg = X[:, None, 1:] * GL_x[None, :, None] + a[None, None, 1:] * (1 - GL_x[None, :, None])
        D = np.linalg.norm(X[:, 1:] - a[1:], axis=1)
        return (1 + gam) * D * (np.abs(Phi(seg)) @ GL_w)
    return Phi, light_excess
def boost_matrix(n, v):
    g = 1 / math.sqrt(1 - v * v); B = np.eye(4)
    B[0, 0] = g; B[0, 1:] = g * v * n; B[1:, 0] = g * v * n; B[1:, 1:] += (g - 1) * np.outer(n, n); return B
def one(p, gam, kind, rs):
    Phi, lex = make(p, gam)
    nhat = rs.normal(size=3); nhat /= np.linalg.norm(nhat); c = np.concatenate([[0.0], RC * nhat])
    if kind == 'покой': n, v = np.array([1.0, 0, 0]), 0.0
    elif kind == 'вдоль': n, v = nhat, V
    else: w = np.cross(nhat, rs.normal(size=3)); n, v = w / np.linalg.norm(w), V
    B = boost_matrix(n, v); u = B[:, 0].copy(); ph = Phi(c[1:][None])[0]
    u[0] /= math.sqrt(1 + 2 * ph); u[1:] /= math.sqrt(1 - 2 * gam * ph)
    p0 = c - TAU * u / 2; q0 = c + TAU * u / 2; H = 1.07 * TAU
    ext = 0.55 * H * ((G * V + 1) if v > 0 else 1.0)
    sqmax = 1 + abs(1 - 3 * gam) * AMP * (RC / (RC - ext)) ** p * 1.05
    n_pts = rs.poisson(RHO0 * sqmax * (math.pi / 24) * H ** 4)
    Z = []
    while sum(len(z) for z in Z) < n_pts:
        Y = rs.uniform(-H / 2, H / 2, (8 * n_pts + 100, 4)); Z.append(Y[np.abs(Y[:, 0]) + np.linalg.norm(Y[:, 1:], axis=1) <= H / 2])
    X = np.concatenate(Z)[:n_pts] @ B.T + c
    sq = 1 + (1 - 3 * gam) * Phi(X[:, 1:])
    X = X[rs.random(len(X)) < sq / sqmax]
    d1 = X - p0; ok = d1[:, 0] > np.linalg.norm(d1[:, 1:], axis=1) + lex(p0, X)
    d2 = q0 - X; ok &= d2[:, 0] > np.linalg.norm(d2[:, 1:], axis=1) + lex(q0, X)
    s = (np.arange(200) + 0.5) / 200; pts = p0[None] + s[:, None] * (q0 - p0)[None]
    phs = Phi(pts[:, 1:]); dt = q0[0] - p0[0]; dx2 = ((q0[1:] - p0[1:]) ** 2).sum()
    tau = np.mean(np.sqrt((1 + 2 * phs) * dt * dt - (1 - 2 * gam * phs) * dx2))
    V0 = RHO0 * math.pi / 24 * tau ** 4
    return ok.sum() / V0 - 1
if __name__ == '__main__':
    t0 = time.time()
    print(f'Φ(r₀=10) = −0.03, ромбы высоты {TAU}, буст {V}, по {NDIA} ромбов')
    print(f"{'p':>5}{'γ':>5} {'δ покой':>16} {'анизотропия':>18}")
    PS = [float(x) for x in os.environ.get('PS', '0.5,1,2').split(',')]
    GS = [float(x) for x in os.environ.get('GS', '-1,1').split(',')]
    KINDS = os.environ.get('KINDS', 'покой,вдоль,поперёк').split(',')
    for p in PS:
        for gam in GS:
            res = {}
            for kind in KINDS:
                rs = np.random.default_rng(int(100 * p) * 10 + int(gam + 3) + {'покой': 1, 'вдоль': 2, 'поперёк': 3}[kind] * 1000)
                d = np.array([one(p, gam, kind, rs) for _ in range(NDIA)])
                res[kind] = (d.mean(), d.std(ddof=1) / math.sqrt(len(d)))
            r0m, r0e = res['покой']
            if 'вдоль' in res:
                an = res['вдоль'][0] - res['поперёк'][0]; ae = math.hypot(res['вдоль'][1], res['поперёк'][1]); ans = f'{an:>+9.5f}±{ae:.5f} ({an/ae:+5.1f}σ)'
            else: ans = '—'
            print(f'{p:>5}{gam:>5.0f} {r0m:>+9.5f}±{r0e:.5f} ({r0m/r0e:+5.1f}σ) {ans}   [{time.time()-t0:.0f} с]')
            sys.stdout.flush()
