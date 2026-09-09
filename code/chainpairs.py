"""Пары цепей: мировые линии и их взаимодействие.

Цепь — последовательность прямо связанных событий. Мы нашли, что цепи
выделены: их связность вдвое выше случайных наборов и падает с расстоянием.
Здесь разбираем это подробно.
"""
import numpy as np


def links_of(R):
    A = np.asarray(R, dtype=np.float32)
    return np.asarray(R & ((A @ A) == 0), dtype=bool)


def depth_of(L, N):
    ind = L.sum(axis=1).astype(np.int64)
    D = np.zeros(N, dtype=np.int64)
    rd = list(np.flatnonzero(ind == 0))
    while rd:
        j = rd.pop(); kd = np.flatnonzero(L[:, j])
        if kd.size:
            D[kd] = np.maximum(D[kd], D[j]+1); ind[kd] -= 1
            rd.extend(kd[ind[kd] == 0].tolist())
    return D


def grow_chain(L, start, rs, maxlen=30):
    cur = start; ch = [start]
    for _ in range(maxlen):
        nxt = np.flatnonzero(L[:, cur])
        if len(nxt) == 0:
            break
        cur = int(rs.choice(nxt)); ch.append(cur)
    return ch


def chain_set(L, D, rs, want=30, minlen=4):
    starts = np.flatnonzero(D == 0)
    out = []
    for _ in range(want*30):
        if not len(starts):
            break
        ch = grow_chain(L, int(rs.choice(starts)), rs)
        if len(ch) >= minlen:
            out.append(ch)
        if len(out) >= want:
            break
    return out


def pair_features(X, R, L, c1, c2):
    """Признаки пары цепей."""
    a = np.array(c1); b = np.array(c2)
    fwd = int(R[np.ix_(b, a)].sum()); bwd = int(R[np.ix_(a, b)].sum())
    lf = int(L[np.ix_(b, a)].sum()) + int(L[np.ix_(a, b)].sum())
    shared = len(set(c1) & set(c2))
    d = float(np.linalg.norm(X[a].mean(0) - X[b].mean(0)))
    # угол между направлениями цепей
    v1 = X[a[-1]] - X[a[0]]; v2 = X[b[-1]] - X[b[0]]
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    cos = float(v1 @ v2/max(n1*n2, 1e-12))
    return dict(dist=d, rel=(fwd+bwd)/(len(a)*len(b)),
                links=lf/(len(a)*len(b)), shared=shared,
                cos=cos, asym=(fwd-bwd)/max(fwd+bwd, 1),
                len1=len(a), len2=len(b))
