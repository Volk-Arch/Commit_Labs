"""Золотая решётка 1+1 против засева: те же различители, что и всегда."""
import sys, numpy as np
sys.path.insert(0, '/home/claude')
from core import frac_pairs, mm_dim, height, links_of, interval_counts, spectral
from goldlat import gold_points
def feats(u, v):
    R = (u[None, :] < u[:, None]) & (v[None, :] < v[:, None]); N = len(u)
    f = frac_pairs(R); n = interval_counts(R, 4)
    return [N, f, mm_dim(f), height(R)[0], n[0]/N, n[1]/n[0], n[2]/n[0], n[3]/n[0], spectral(R)]
names = ['N', 'доля', 'разм', 'высота', 'связ/соб', 'n1/n0', 'n2/n0', 'n3/n0', 'спектр']
for U in (20, 32):
    u, v = gold_points(float(U)); g = feats(u, v); N = len(u)
    P = np.array([feats(*np.random.default_rng(s).uniform(0, U, (2, N))) for s in range(8)])
    print(f'U={U}')
    for k, nm in enumerate(names):
        print(f'   {nm:9s} золотая {g[k]:9.4f}   засев {P[:,k].mean():9.4f} ± {P[:,k].std(ddof=1):.4f}   z={(g[k]-P[:,k].mean())/max(P[:,k].std(ddof=1),1e-12):+6.1f}')
