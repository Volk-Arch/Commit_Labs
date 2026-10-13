import numpy as np
import sys; sys.path.insert(0, '/home/claude')
from cliff import clifford, transposition_op

def lift_along(P, g, k):
    """Подъём вдоль пути: на каждой смене ранжировки — транспозиция."""
    d = g[0].shape[0]
    M = np.eye(d, dtype=complex)
    perms = [tuple(np.argsort(row)) for row in P]
    for t in range(1, len(perms)):
        a, b = perms[t-1], perms[t]
        pa = list(a)
        for i in range(k):
            if pa[i] != b[i]:
                j = pa.index(b[i])
                M = M @ transposition_op(g, i, j)
                pa[i], pa[j] = pa[j], pa[i]
    return M

def exchange_path(pA, pB, steps=2000, k=4):
    """Две копии обмениваются местами: A идёт по полукругу к B, B к A."""
    c = 0.5*(pA+pB); r = 0.5*np.linalg.norm(pB-pA)
    d = (pB-pA)/np.linalg.norm(pB-pA)
    # ортогональное направление
    e = np.random.default_rng(7).normal(size=k)
    e -= (e@d)*d; e /= np.linalg.norm(e)
    th = np.linspace(0, np.pi, steps)
    A = c[None, :] + r*(np.cos(th)[:, None]*(-d)[None, :] + np.sin(th)[:, None]*e[None, :])
    B = c[None, :] + r*(np.cos(th)[:, None]*( d)[None, :] - np.sin(th)[:, None]*e[None, :])
    return np.mod(A, 1.0), np.mod(B, 1.0)
