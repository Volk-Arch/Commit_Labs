import numpy as np
import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from slice import antichain_slice, slice_structure


def pt(K):
    Kb = np.ascontiguousarray(K, dtype=bool); n = len(Kb)
    ind = Kb.sum(axis=1).astype(np.int64); D = np.zeros(n, dtype=np.int64)
    rd = list(np.flatnonzero(ind == 0))
    while rd:
        j = rd.pop(); kd = np.flatnonzero(Kb[:, j])
        if kd.size:
            D[kd] = np.maximum(D[kd], D[j]+1); ind[kd] -= 1
            rd.extend(kd[ind[kd] == 0].tolist())
    return D


def find_groups(R, idx, th, lo=3, frac=3):
    """Группы на срезе: связные компоненты по близости выше порога."""
    prox, _ = slice_structure(R, idx)
    G = prox > th
    n = len(idx)
    seen = np.zeros(n, bool); out = []
    for s in range(n):
        if seen[s]: continue
        st = [s]; c = []
        while st:
            v = st.pop()
            if seen[v]: continue
            seen[v] = True; c.append(v)
            st.extend(np.flatnonzero(G[v] & ~seen).tolist())
        if lo <= len(c) <= max(lo, n//frac):
            out.append(np.array([idx[t] for t in c]))
    del prox
    return out
