#!/usr/bin/env python3
"""Та же «одна формула», но мир построен ТОЧНО для метрики
    ds² = −e^{2Φ}dt² + e^{−2γΦ}dx²,   Φ = −A·(r₀/r)^p:
густота √−g = e^{(1−3γ)Φ}, время света ∫e^{−(1+γ)Φ}dl, местная рамка и
собственное время — через экспоненты. Остаётся одна неточность: свет идёт по
прямой, а не по геодезической (поправка второго порядка от изгиба).
Проверка: даёт ли постоянный потенциал (p = 0) и поле 1/r (p = 1) нуль БЕЗ
общего сдвига, который был у построения первого порядка (+0.0009 при Φ = 0.01)."""
import os, sys, math, time
import numpy as np
RC, TAU, V, RHO0 = float(os.environ.get('RC', 12)), float(os.environ.get('TAU', 8)), 0.8, float(os.environ.get('RHO0', 12))
AMP = float(os.environ.get('AMP', 0.03))
G = 1 / math.sqrt(1 - V * V)
GL_x, GL_w = np.polynomial.legendre.leggauss(8); GL_x = 0.5 * (GL_x + 1); GL_w = 0.5 * GL_w
def boost_matrix(n, v):
    g = 1 / math.sqrt(1 - v * v); B = np.eye(4)
    B[0, 0] = g; B[0, 1:] = g * v * n; B[1:, 0] = g * v * n; B[1:, 1:] += (g - 1) * np.outer(n, n); return B
def one(p, gam, kind, rs):
    Phi = lambda X: -AMP * (RC / np.maximum(np.linalg.norm(X, axis=-1), 1e-6)) ** p
    def light(a, X):
        seg = X[:, None, 1:] * GL_x[None, :, None] + a[None, None, 1:] * (1 - GL_x[None, :, None])
        D = np.linalg.norm(X[:, 1:] - a[1:], axis=1)
        return D * (np.exp(-(1 + gam) * Phi(seg)) @ GL_w)
    nhat = rs.normal(size=3); nhat /= np.linalg.norm(nhat); c = np.concatenate([[0.0], RC * nhat])
    if kind == 'покой': n, v = np.array([1.0, 0, 0]), 0.0
    elif kind == 'вдоль': n, v = nhat, V
    else: w = np.cross(nhat, rs.normal(size=3)); n, v = w / np.linalg.norm(w), V
    B = boost_matrix(n, v); u = B[:, 0].copy(); ph = Phi(c[1:][None])[0]
    u[0] *= math.exp(-ph); u[1:] *= math.exp(gam * ph)
    p0 = c - TAU * u / 2; q0 = c + TAU * u / 2; H = 1.08 * TAU
    ext = 0.55 * H * ((G * V + 1) if v > 0 else 1.0)
    phmax = AMP * (RC / (RC - ext)) ** p
    sqmax = math.exp(abs(1 - 3 * gam) * phmax) * 1.01
    S = np.diag([math.exp(-ph)] + [math.exp(gam * ph)] * 3); jac = np.linalg.det(S)
    n_pts = rs.poisson(RHO0 * sqmax * jac * (math.pi / 24) * H ** 4)
    Z = []
    while sum(len(z) for z in Z) < n_pts:
        Y = rs.uniform(-H / 2, H / 2, (8 * n_pts + 100, 4)); Z.append(Y[np.abs(Y[:, 0]) + np.linalg.norm(Y[:, 1:], axis=1) <= H / 2])
    Y = np.concatenate(Z)[:n_pts]
    X = Y @ (S @ B).T + c                       # координатная густота = RHO0·sqmax
    X = X[rs.random(len(X)) < np.exp((1 - 3 * gam) * Phi(X[:, 1:])) / sqmax]
    d1 = X - p0; ok = d1[:, 0] > light(p0, X)
    d2 = q0 - X; ok &= d2[:, 0] > light(q0, X)
    s = (np.arange(200) + 0.5) / 200; pts = p0[None] + s[:, None] * (q0 - p0)[None]
    phs = Phi(pts[:, 1:]); dt = q0[0] - p0[0]; dx2 = ((q0[1:] - p0[1:]) ** 2).sum()
    tau = np.mean(np.sqrt(np.exp(2 * phs) * dt * dt - np.exp(-2 * gam * phs) * dx2))
    return ok.sum() / (RHO0 * math.pi / 24 * tau ** 4) - 1
if __name__ == '__main__':
    NDIA = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    t0 = time.time()
    for spec in os.environ.get('CASES', '0:1:покой,1:1:покой,2:1:покой').split(','):
        p, gam, kind = spec.split(':'); p, gam = float(p), float(gam)
        rs = np.random.default_rng(int(p * 100) + int((gam + 3) * 7) + hash(kind) % 1000)
        d = np.array([one(p, gam, kind, rs) for _ in range(NDIA)])
        print(f'p={p} γ={gam:+.0f} {kind:8s}: δ = {d.mean():+.5f} ± {d.std(ddof=1)/math.sqrt(NDIA):.5f}   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
