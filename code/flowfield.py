import numpy as np
import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from linkstate import rel_of, links_of


def flow_field(X):
    """Поток на связях: сохраняется на каждом узле точно.
    Истоки создают единицу; каждое событие делит полученное поровну."""
    N = len(X)
    R = rel_of(X); L = links_of(R); Lb = np.asarray(L, dtype=bool)
    ind = Lb.sum(axis=1).astype(np.int64)
    rd = list(np.flatnonzero(ind == 0)); order = []
    while rd:
        j = rd.pop(); order.append(j)
        kd = np.flatnonzero(Lb[:, j])
        if kd.size:
            ind[kd] -= 1; rd.extend(kd[ind[kd] == 0].tolist())
    node = np.zeros(N)
    node[np.flatnonzero(Lb.sum(axis=1) == 0)] = 1.0
    for v in order:
        kids = np.flatnonzero(Lb[:, v])
        if len(kids) == 0: continue
        node[kids] += node[v]/len(kids)
    return R, L, node


def neigh_frac(R, L, i, minn=20):
    ng = np.flatnonzero(R[i] | R[:, i])
    if len(ng) < minn: return np.nan
    sub = R[np.ix_(ng, ng)]; subL = L[np.ix_(ng, ng)]
    t = float(sub.sum())
    return float(subL.sum()/t) if t > 0 else np.nan
