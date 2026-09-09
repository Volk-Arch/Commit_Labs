import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
from itertools import combinations
exec(open('/home/claude/coarse.py').read())

def sector_ensemble(U, size):
    """Все сектора заданного размера: подмножества шкал."""
    k = U.shape[0]
    out = []
    for s in combinations(range(k), size):
        R = np.ones((U.shape[1],)*2, dtype=bool)
        for m in s: R &= (U[m][None, :] < U[m][:, None])
        np.fill_diagonal(R, False)
        out.append((s, R))
    return out

def typicality(sectors):
    """Насколько каждый сектор «типичен»: средняя согласованность с остальными."""
    n = len(sectors)
    M = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b: M[a, b] = 1.0; continue
            i = (sectors[a][1] & sectors[b][1]).sum()
            u = (sectors[a][1] | sectors[b][1]).sum()
            M[a, b] = i/max(u, 1)
    typ = (M.sum(axis=1)-1)/(n-1)
    return M, typ
