import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/order6.py').read().split("if __name__")[0])
from collections import Counter
rng8 = np.random.default_rng(101)

def twins_connected(K, minlink=3):
    """Близнецы СРЕДИ СВЯЗАННЫХ: одинаковое прошлое и будущее, и оно непусто."""
    Kb = np.ascontiguousarray(np.asarray(K, dtype=bool)); n = len(Kb)
    live = (Kb.sum(axis=1) >= minlink) & (Kb.sum(axis=0) >= minlink)
    sig = {}
    for i in np.flatnonzero(live):
        sig.setdefault((Kb[i].tobytes(), Kb[:, i].tobytes()), []).append(i)
    pairs = sum(len(v)*(len(v)-1)//2 for v in sig.values())
    grp = sum(1 for v in sig.values() if len(v) > 1)
    return pairs, grp, int(live.sum())

def clamped(N, k, levels, rs):
    """Зажать одну шкалу до levels уровней — совпадения фаз становятся возможны."""
    U = [rs.permutation(N) for _ in range(k)]
    U[0] = (U[0]*levels)//N
    R = np.ones((N, N), dtype=bool)
    for u in U: R &= (u[None, :] < u[:, None])
    return R.astype(np.float32)

def defective(N, k, frac, rs):
    """Доля frac событий получает КОПИЮ фаз другого события — «дефектные»."""
    U = np.array([rs.permutation(N) for _ in range(k)], dtype=np.int64)
    m = int(frac*N)
    if m > 0:
        src = rs.choice(N, m, replace=False)
        dst = rs.choice(np.setdiff1d(np.arange(N), src), m, replace=False)
        U[:, dst] = U[:, src]
    R = np.ones((N, N), dtype=bool)
    for u in U: R &= (u[None, :] < u[:, None])
    return R.astype(np.float32)
