"""Причинный порядок из истории коммитов — настоящие данные."""
import numpy as np


def load_commits(path):
    par = {}
    for line in open(path):
        p = line.split()
        if p:
            par[p[0]] = p[1:]
    return par


def sample_order(par, n, rs, window=None):
    """Берём связный кусок истории и строим порядок на нём."""
    ids = list(par)
    # берём последовательный отрезок — это связный кусок истории
    start = int(rs.integers(0, max(1, len(ids)-n)))
    sub = ids[start:start+n]
    idx = {c: i for i, c in enumerate(sub)}
    m = len(sub)
    # прямые связи
    L = np.zeros((m, m), dtype=bool)
    for c in sub:
        for p in par.get(c, []):
            if p in idx:
                L[idx[c], idx[p]] = True
    # транзитивное замыкание
    R = L.copy()
    for _ in range(200):
        Rn = R | ((R.astype(np.float32) @ R.astype(np.float32)) > 0)
        if Rn.sum() == R.sum():
            break
        R = Rn
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)
