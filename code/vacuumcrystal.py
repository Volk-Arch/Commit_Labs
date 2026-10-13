"""Кто гнёт: правило вакуума на КРИСТАЛЛЕ вместо засева (повтор vacuum.py).
Семейство ds² = −(1+2Φ)dt² + (1−2γΦ)dx², Φ = −M/r. «Кто раньше» — как в vacuum.py.
«Кто есть»: золотой кристалл в общей системе относительно ромба (случайный буст 0.6 и поворот),
густота ∝ √−g задаётся радиусом окна тени в каждой точке: r(X) = r_max·(√−g(X)/max)^(1/4) —
детерминированно, без прореживания. Анизотропия δ(вдоль) − δ(поперёк) ∝ (γ − 1):
нуль только у Эйнштейна. Кристалл шумит меньше засева ⇒ γ* должен определиться точнее."""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
from scipy.linalg import expm
from hidden import points_lab
from gold4 import r_for_density
from vacuum import M, RC, TAU, V, G, Phi, proper, boost_matrix
RHO0 = 30.0; NDIA = int(sys.argv[1]) if len(sys.argv) > 1 else 200
GAMS = [float(g) for g in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['-1', '1'])]
def earlier_pts(a, X, gam, sign=1):
    d = (X - a) * sign; D = np.linalg.norm(d[:, 1:], axis=1)
    ra = np.linalg.norm(a[1:]); rx = np.linalg.norm(X[:, 1:], axis=1)
    S = np.log((ra + rx + D) / np.maximum(ra + rx - D, 1e-12))
    return d[:, 0] > D + (1 + gam) * M * S
def one(gam, kind, rs, crystal=True):
    nhat = rs.normal(size=3); nhat /= np.linalg.norm(nhat); c = np.concatenate([[0.0], RC * nhat])
    w = np.cross(nhat, rs.normal(size=3)); w /= np.linalg.norm(w)
    n = nhat if kind == 'вдоль r' else w
    B = boost_matrix(n, V); u = B[:, 0].copy(); ph_c = -M / RC
    u[0] /= math.sqrt(1 + 2 * ph_c); u[1:] /= math.sqrt(1 - 2 * gam * ph_c)
    p = c - TAU * u / 2; q = c + TAU * u / 2; H = 1.06 * TAU
    sqmax = 1 + abs(1 - 3 * gam) * M / (RC - 0.55 * H * (G * V + 1))
    if crystal:
        # кристалл в своей системе; Lc — его случайный буст 0.6 и поворот относительно ромба
        Gm = np.zeros((4, 4)); bv = rs.normal(size=3); bv *= 0.6 / np.linalg.norm(bv); Gm[0, 1:] = Gm[1:, 0] = bv
        R = np.zeros((4, 4)); r3 = rs.normal(0, 1.5, 3); R[1, 2], R[2, 1], R[1, 3], R[3, 1], R[2, 3], R[3, 2] = r3[0], -r3[0], r3[1], -r3[1], r3[2], -r3[2]
        Lc = expm(Gm) @ expm(R); T = B @ Lc                   # точки: X = T·Y + c
        Ti = np.linalg.inv(T)
        corners = np.array([[a, b_, cc, d] for a in (-H/2, H/2) for b_ in (-H/2, H/2) for cc in (-H/2, H/2) for d in (-H/2, H/2)])
        Yc = corners @ B.T @ Ti.T                              # прообраз куба ромба в системе кристалла (с запасом)
        lo, hi = Yc.min(0), Yc.max(0); sh = rs.uniform(0, 400, 4)
        rmax = r_for_density(RHO0 * sqmax)
        Y, s, _ = points_lab([(lo[k] + sh[k], hi[k] + sh[k]) for k in range(4)], rmax)
        X = (Y - sh) @ T.T + c
        sq = 1 + (1 - 3 * gam) * Phi(X[:, 1:])
        if __import__('os').environ.get('NOMOD'): X = X[s <= rmax * (1 / sqmax) ** 0.25]
        else: X = X[s <= rmax * (sq / sqmax) ** 0.25]
    else:
        nexp = RHO0 * sqmax * H ** 4 * 4                        # засев в кубе вокруг центра
        Y = rs.uniform(-H, H, (rs.poisson(RHO0 * sqmax * (2 * H) ** 4), 4)); X = Y + c
        sq = 1 + (1 - 3 * gam) * Phi(X[:, 1:]); X = X[rs.random(len(X)) < sq / sqmax]
    inside = earlier_pts(p, X, gam) & earlier_pts(q, X, gam, sign=-1)
    N = inside.sum(); tau = proper(p, q, gam); V0 = RHO0 * math.pi / 24 * tau ** 4
    return N / V0 - 1
if __name__ == '__main__':
    kind_world = sys.argv[3] if len(sys.argv) > 3 else 'crystal'
    SOFF = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    t0 = time.time(); res = {}
    for gam in GAMS:
        for kind in ('вдоль r', 'поперёк'):
            rs = np.random.default_rng(int(1000 * (gam + 3)) + (2 if kind == 'вдоль r' else 3) + (0 if kind_world == 'crystal' else 77) + 100000 * SOFF)
            d = np.array([one(gam, kind, rs, kind_world == 'crystal') for _ in range(NDIA)])
            res[(gam, kind)] = (d.mean(), d.std(ddof=1) / math.sqrt(NDIA), d.std(ddof=1))
            print(f'{kind_world} γ={gam:+.0f} {kind:8s}: δ = {d.mean():+.5f} ± {res[(gam,kind)][1]:.5f} (разброс на ромб {d.std(ddof=1):.4f})   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
    gs, ds_, es = [], [], []
    for gam in GAMS:
        diff = res[(gam, 'вдоль r')][0] - res[(gam, 'поперёк')][0]; err = math.hypot(res[(gam, 'вдоль r')][1], res[(gam, 'поперёк')][1])
        th = (gam - 1) * (G * V) ** 2 * (-3 * M / RC ** 3) * TAU ** 2 / 30
        print(f'   γ={gam:+.0f}: анизотропия {diff:+.5f} ± {err:.5f}  (теория {th:+.5f})'); gs.append(gam); ds_.append(diff); es.append(err)
    if len(gs) >= 2:
        gs, ds_, es = map(np.array, (gs, ds_, es)); w = 1 / es ** 2
        A = np.vstack([np.ones_like(gs), gs]).T; cov = np.linalg.inv(A.T @ (A * w[:, None])); a, b = cov @ (A.T @ (w * ds_))
        gstar = -a / b; egs = abs(gstar) * math.sqrt(cov[0, 0] / a ** 2 + cov[1, 1] / b ** 2 - 2 * cov[0, 1] / (a * b))
        print(f'   {kind_world}: нуль анизотропии при γ* = {gstar:.2f} ± {egs:.2f}  (Эйнштейн 1, Нордстрём −1)')
