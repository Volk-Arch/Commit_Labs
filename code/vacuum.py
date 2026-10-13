#!/usr/bin/env python3
"""ОТКУДА ДВОЙКА: вакуум как правило о малых ромбах.

Семейство слабых полей:  ds² = −(1+2Φ)dt² + (1−2γΦ)dx²,  Φ = −M/r.
γ = 1 — Эйнштейн (свет: 1+γ = 2), γ = −1 — Нордстрём (свет: 0).
Мир строится обеими строками: «кто есть» — густота ∝ √−g = 1 + Φ − 3γΦ;
«кто раньше» — x ≻ a ⟺ Δt > |Δx| + (1+γ)·M·ln((r_a+r_x+D)/(r_a+r_x−D))
(время света в первом порядке; в tilt-doc этот перенос проверен замыканием).

ПРАВИЛО ВАКУУМА: вне вещества малый причинный ромб любой ориентации
содержит столько событий, сколько плоский той же собственной высоты.
Измеряем δ = N/(ρ₀·π τ⁴/24) − 1 для ромбов в покое, с бустом вдоль радиуса
и поперёк, на расстоянии r = 10M. Теория (Гиббонс–Солодухин):
δ = −R τ²/180 + R_uu τ²/30, вне источника R = 0, R_uu = (γ−1)·γ²v²·∂ₙ²Φ:
вдоль радиуса ∂²Φ = −2M/r³, поперёк +M/r³. Нуль при ВСЕХ ориентациях — только γ = 1.
"""
import sys, math, time
import numpy as np
M, RC, TAU, V, RHO0, NDIA = 0.1, 10.0, 6.0, 0.8, 30.0, int(sys.argv[1]) if len(sys.argv) > 1 else 300
G = 1 / math.sqrt(1 - V * V)
def Phi(X): return -M / np.linalg.norm(X, axis=-1)
def earlier(a, X, gam):
    """a ≺ x для точек X"""
    d = X - a; D = np.linalg.norm(d[:, 1:], axis=1)
    ra = np.linalg.norm(a[1:]); rx = np.linalg.norm(X[:, 1:], axis=1)
    S = np.log((ra + rx + D) / np.maximum(ra + rx - D, 1e-12))
    return d[:, 0] > D + (1 + gam) * M * S
def proper(p, q, gam, k=200):
    s = (np.arange(k) + 0.5) / k; pts = p[None] + s[:, None] * (q - p)[None]
    ph = Phi(pts[:, 1:]); dt = q[0] - p[0]; dx2 = ((q[1:] - p[1:]) ** 2).sum()
    return np.mean(np.sqrt((1 + 2 * ph) * dt * dt - (1 - 2 * gam * ph) * dx2))
def boost_matrix(n, v):
    g = 1 / math.sqrt(1 - v * v); B = np.eye(4)
    B[0, 0] = g; B[0, 1:] = g * v * n; B[1:, 0] = g * v * n
    B[1:, 1:] += (g - 1) * np.outer(n, n); return B
def one(gam, kind, rs):
    nhat = rs.normal(size=3); nhat /= np.linalg.norm(nhat); c = np.concatenate([[0.0], RC * nhat])
    if kind == 'покой': n, v = np.array([1.0, 0, 0]), 0.0
    elif kind == 'вдоль r': n, v = nhat, V
    else:
        w = np.cross(nhat, rs.normal(size=3)); n, v = w / np.linalg.norm(w), V
    B = boost_matrix(n, v); u = B[:, 0].copy()
    ph_c = -M / RC                                  # местная ортонормированная рамка в центре
    u[0] /= math.sqrt(1 + 2 * ph_c); u[1:] /= math.sqrt(1 - 2 * gam * ph_c)
    p = c - TAU * u / 2; q = c + TAU * u / 2
    H = 1.06 * TAU
    sqmax = 1 + abs(1 - 3 * gam) * M / (RC - 0.55 * H * (G * V + 1))
    nexp = RHO0 * sqmax * (math.pi / 24) * H ** 4
    n_pts = rs.poisson(nexp)
    Z = []
    while sum(len(z) for z in Z) < n_pts:
        Y = rs.uniform(-H / 2, H / 2, (8 * n_pts + 100, 4)); Y = Y[np.abs(Y[:, 0]) + np.linalg.norm(Y[:, 1:], axis=1) <= H / 2]; Z.append(Y)
    Y = np.concatenate(Z)[:n_pts]
    X = Y @ B.T + c
    sq = 1 + (1 - 3 * gam) * Phi(X[:, 1:])
    X = X[rs.random(len(X)) < sq / sqmax]
    inside = earlier(p, X, gam)
    # q ≻ x  ⟺  x ≺ q: используем симметрию формулы
    d = q - X; D = np.linalg.norm(d[:, 1:], axis=1)
    rq = np.linalg.norm(q[1:]); rx = np.linalg.norm(X[:, 1:], axis=1)
    S = np.log((rq + rx + D) / np.maximum(rq + rx - D, 1e-12))
    inside &= d[:, 0] > D + (1 + gam) * M * S
    N = inside.sum(); tau = proper(p, q, gam)
    V0 = RHO0 * math.pi / 24 * tau ** 4
    return N / V0 - 1, 1 / math.sqrt(V0)
if __name__ == '__main__':
    t0 = time.time()
    print(f'r = {RC}M, Φ = {-M/RC}, высота ромба τ = {TAU}, буст v = {V}; ромбов на строку {NDIA}')
    print(f"{'γ':>4} {'ориентация':>10} {'δ замер':>12} {'±':>8} {'теория R_uu τ²/30':>18}")
    res = {}
    for gam in (-1.0, 0.0, 1.0, 2.0):
        for kind in (('покой', 'вдоль r', 'поперёк') if gam in (-1.0, 1.0) else ('вдоль r', 'поперёк')):
            rs = np.random.default_rng(int(1000 * (gam + 3)) + {'покой': 1, 'вдоль r': 2, 'поперёк': 3}[kind])
            d = np.array([one(gam, kind, rs) for _ in range(NDIA)])
            m, e = d[:, 0].mean(), d[:, 0].std(ddof=1) / math.sqrt(len(d))
            d2 = {'покой': 0.0, 'вдоль r': -2 * M / RC ** 3, 'поперёк': M / RC ** 3}[kind]
            th = (gam - 1) * (G * V if kind != 'покой' else 0) ** 2 * d2 * TAU ** 2 / 30
            res[(gam, kind)] = (m, e)
            print(f'{gam:>4.0f} {kind:>10} {m:>+12.5f} {e:>8.5f} {th:>+18.5f}   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
    print('\nПроверка «вакуума»: при каком γ ромбы всех ориентаций совпадают с плоскими?')
    gs, ds_, es = [], [], []
    for gam in (-1.0, 0.0, 1.0, 2.0):
        diff = res[(gam, 'вдоль r')][0] - res[(gam, 'поперёк')][0]
        err = math.hypot(res[(gam, 'вдоль r')][1], res[(gam, 'поперёк')][1])
        th = (gam - 1) * (G * V) ** 2 * (-3 * M / RC ** 3) * TAU ** 2 / 30
        print(f'   γ={gam:+.0f}:  δ(вдоль) − δ(поперёк) = {diff:+.5f} ± {err:.5f}   ({diff/err:+.1f}σ)   теория {th:+.5f}')
        gs.append(gam); ds_.append(diff); es.append(err)
    gs, ds_, es = map(np.array, (gs, ds_, es)); w = 1 / es ** 2
    A = np.vstack([np.ones_like(gs), gs]).T; cov = np.linalg.inv(A.T @ (A * w[:, None])); a, b = cov @ (A.T @ (w * ds_))
    gstar = -a / b; egs = abs(gstar) * math.sqrt(cov[0, 0] / a ** 2 + cov[1, 1] / b ** 2 - 2 * cov[0, 1] / (a * b))
    print(f'\n   прямая по γ: анизотропия = {a:+.5f} {b:+.5f}·γ;  обращается в нуль при γ* = {gstar:.2f} ± {egs:.2f}   (ОТО: 1, Нордстрём: −1)')
