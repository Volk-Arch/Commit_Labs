import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
from scipy.linalg import solve_triangular
exec(open('/home/claude/tsymm.py').read())

def field_from(R, src, alpha):
    """Поле от источника src вперёд по порядку."""
    Rf = np.ascontiguousarray(np.asarray(R, dtype=np.float32))
    n = len(Rf)
    M = np.eye(n, dtype=np.float32) + alpha*Rf
    return 0.5*solve_triangular(M, Rf[:, src].copy(), lower=True, unit_diagonal=True)

def profile(R, G, src, nb=12):
    """Профиль поля по глубине от источника."""
    D = proper_time(np.asarray(R, dtype=np.float32)).astype(float)
    d0 = D[src]; rel = D - d0
    hi = rel.max()
    if hi < 6: return None
    e = np.linspace(1, hi, nb+1); P = []
    for lo, up in zip(e[:-1], e[1:]):
        s = (rel >= lo) & (rel < up)
        P.append(G[s].mean() if s.sum() > 8 else np.nan)
    return np.array(P)
