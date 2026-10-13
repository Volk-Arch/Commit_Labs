import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/order8.py').read().split("if __name__")[0])
rng14 = np.random.default_rng(191)

def rel_majority(U, need):
    """Ослабленное правило: i раньше j, если раньше хотя бы по `need` шкалам из k."""
    k, N = U.shape
    cnt = np.zeros((N, N), dtype=np.int16)
    for u in U:
        cnt += (u[None, :] < u[:, None]).astype(np.int16)   # cnt[j,i] по скольким i<j
    return (cnt >= need)

def acyclic(R):
    """Есть ли циклы: пробуем топологическую сортировку."""
    Rb = np.ascontiguousarray(R, dtype=bool); n = len(Rb)
    indeg = Rb.sum(axis=1).astype(np.int64)
    ready = list(np.flatnonzero(indeg == 0)); seen = 0
    while ready:
        j = ready.pop(); seen += 1
        kids = np.flatnonzero(Rb[:, j])
        if kids.size:
            indeg[kids] -= 1
            ready.extend(kids[indeg[kids] == 0].tolist())
    return seen == n

def transitive(R, samples=200000, rs=None):
    """Доля троек i<j<k, где транзитивность НЕ нарушена."""
    r = rs if rs is not None else rng14
    Rb = np.ascontiguousarray(R, dtype=bool); n = len(Rb)
    ok = 0; tot = 0
    for _ in range(samples):
        a, b, c = (int(v) for v in r.integers(0, n, 3))
        if Rb[b, a] and Rb[c, b]:
            tot += 1; ok += bool(Rb[c, a])
    return ok/max(tot, 1), tot

def mirror_diff(U, need, i=0, j=1):
    A = rel_majority(U, need)
    V = U.copy(); V[[i, j]] = V[[j, i]]
    B = rel_majority(V, need)
    return int((A ^ B).sum())
