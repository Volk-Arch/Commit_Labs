"""Кто гнёт — правило вакуума на ДЕФОРМИРОВАННОМ кристалле. Густота и геометрия задаются не
радиусом окна, а деформацией: один золотой кристалл режется на кубические лоскуты со стороной s,
каждый лоскут сдвигается к своему центру c и растягивается местной рамкой
E(c) = diag(1/√(1+2Φ), 1/√(1−2γΦ) ×3). Густота тогда сама ∝ √−g; кривизна — в несовпадении рамок
соседних лоскутов (швы-дефекты). Анизотропия δ(вдоль) − δ(поперёк): у Эйнштейна должна исчезнуть."""
import sys, math, time, os, numpy as np
sys.path.insert(0, '.')
from scipy.linalg import expm
from hidden import points_lab
from gold4 import r_for_density
from vacuum import M, RC, TAU, V, G, Phi, proper, boost_matrix
from vacuumcrystal import earlier_pts
RHO0 = 30.0; NDIA = int(sys.argv[1]); GAMS = [float(g) for g in sys.argv[2].split(',')]; S = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
SOFF = int(sys.argv[4]) if len(sys.argv) > 4 else 0
KEEPALL = len(sys.argv) > 5 and sys.argv[5] == 'all'
def one(gam, kind, rs):
    nhat = rs.normal(size=3); nhat /= np.linalg.norm(nhat); c = np.concatenate([[0.0], RC * nhat])
    w = np.cross(nhat, rs.normal(size=3)); w /= np.linalg.norm(w); n = nhat if kind == 'вдоль r' else w
    B = boost_matrix(n, V); u = B[:, 0].copy(); ph_c = -M / RC
    u[0] /= math.sqrt(1 + 2 * ph_c); u[1:] /= math.sqrt(1 - 2 * gam * ph_c)
    p = c - TAU * u / 2; q = c + TAU * u / 2; H = 1.06 * TAU
    Gm = np.zeros((4, 4)); bv = rs.normal(size=3); bv *= 0.6 / np.linalg.norm(bv); Gm[0, 1:] = Gm[1:, 0] = bv
    R = np.zeros((4, 4)); r3 = rs.normal(0, 1.5, 3); R[1, 2], R[2, 1], R[1, 3], R[3, 1], R[2, 3], R[3, 2] = r3[0], -r3[0], r3[1], -r3[1], r3[2], -r3[2]
    T = B @ expm(Gm) @ expm(R); Ti = np.linalg.inv(T)
    corners = np.array([[a, b_, cc, d] for a in (-H/2, H/2) for b_ in (-H/2, H/2) for cc in (-H/2, H/2) for d in (-H/2, H/2)])
    Yc = corners @ B.T @ Ti.T; lo, hi = Yc.min(0) - 0.6, Yc.max(0) + 0.6; sh = rs.uniform(0, 400, 4)
    Y, s, _ = points_lab([(lo[k] + sh[k], hi[k] + sh[k]) for k in range(4)], r_for_density(RHO0))
    Z = (Y - sh) @ T.T + c                                   # недеформированное положение
    org = c - 50.0 + rs.uniform(0, S, 4)                     # случайная сетка лоскутов
    k = np.floor((Z - org) / S); cen = org + (k + 0.5) * S
    ph = Phi(cen[:, 1:])
    E = np.column_stack([1 / np.sqrt(1 + 2 * ph)] + [1 / np.sqrt(1 - 2 * gam * ph)] * 3)
    X = cen + E * (Z - cen)
    if not KEEPALL: X = X[np.all(np.floor((X - org) / S) == k, axis=1)]     # точка осталась в своём лоскуте
    # KEEPALL: швы как есть — нахлёсты и щели, каждый лоскут отображает свой объём целиком
    inside = earlier_pts(p, X, gam) & earlier_pts(q, X, gam, sign=-1)
    N = inside.sum(); tau = proper(p, q, gam); V0 = RHO0 * math.pi / 24 * tau ** 4
    return N / V0 - 1
if __name__ == '__main__':
    t0 = time.time(); res = {}
    for gam in GAMS:
        for kind in ('вдоль r', 'поперёк'):
            rs = np.random.default_rng(int(1000 * (gam + 3)) + (2 if kind == 'вдоль r' else 3) + 555 + 100000 * SOFF)
            d = np.array([one(gam, kind, rs) for _ in range(NDIA)])
            res[(gam, kind)] = (d.mean(), d.std(ddof=1) / math.sqrt(NDIA))
            print(f'лоскуты {S}{" (швы как есть)" if KEEPALL else ""} γ={gam:+.0f} {kind:8s}: δ = {d.mean():+.5f} ± {res[(gam,kind)][1]:.5f} (разброс на ромб {d.std(ddof=1):.4f})   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
    for gam in GAMS:
        diff = res[(gam, 'вдоль r')][0] - res[(gam, 'поперёк')][0]; err = math.hypot(res[(gam, 'вдоль r')][1], res[(gam, 'поперёк')][1])
        th = (gam - 1) * (G * V) ** 2 * (-3 * M / RC ** 3) * TAU ** 2 / 30
        print(f'   γ={gam:+.0f}: анизотропия {diff:+.5f} ± {err:.5f}  (теория {th:+.5f})')
