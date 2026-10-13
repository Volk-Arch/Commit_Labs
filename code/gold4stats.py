"""Золотой квазикристалл 3+1 в причинном ромбе против засева 4d."""
import sys, math, time, numpy as np
sys.path.insert(0, '/home/claude'); sys.path.insert(0, '.')
from gold4 import points_in_box, r_for_density
from drawlab import sprinkle_R
from core import frac_pairs, mm_dim, height, links_of, interval_counts, spectral
V4 = 2 * math.pi / 3
def order(P):
    dt = P[:, None, 0] - P[None, :, 0]; dx = np.linalg.norm(P[:, None, 1:] - P[None, :, 1:], axis=-1)
    R = dt > dx; np.fill_diagonal(R, False); return R
def feats(R):
    N = len(R); f = frac_pairs(R); n = interval_counts(R, 4)
    return np.array([N, f, mm_dim(f), height(R)[0], n[0]/N, n[1]/n[0], n[2]/n[0], n[3]/n[0], spectral(R)])
NM = ['N', 'доля', 'разм', 'высота', 'связ/соб', 'n1/n0', 'n2/n0', 'n3/n0', 'спектр']
rs = np.random.default_rng(4); t0 = time.time()
for Nt in (1000, 2000, 4000):
    r = r_for_density(Nt / V4)
    G = []
    for s in range(3):
        c = rs.uniform(0, 100, 4)
        P = points_in_box([(c[k] - 1, c[k] + 1) for k in range(4)], r) - c
        P = P[np.abs(P[:, 0]) + np.linalg.norm(P[:, 1:], axis=1) <= 1]
        G.append(feats(order(P)))
    G = np.array(G)
    Ng = int(G[:, 0].mean())
    S = np.array([feats(sprinkle_R(Ng, np.random.default_rng(300 + s))) for s in range(6)])
    print(f'N≈{Nt}  [{time.time()-t0:.0f} с]')
    for k, nm in enumerate(NM):
        sd = S[:, k].std(ddof=1)
        z = (G[:, k].mean() - S[:, k].mean()) / sd if sd > 0 else float('nan')
        print(f'   {nm:9s} квазикристалл {G[:,k].mean():9.4f} (±{G[:,k].std():.4f})   засев {S[:,k].mean():9.4f} ± {sd:.4f}   z={z:+6.1f}')
    sys.stdout.flush()
