import numpy as np

def chain_depth(R, idx):
    """Высота подмножества — собственное время внутри интервала."""
    sub = np.ascontiguousarray(R[np.ix_(idx, idx)], dtype=bool)
    n = len(sub)
    ind = sub.sum(axis=1).astype(np.int64)
    D = np.zeros(n, dtype=np.int64)
    rd = list(np.flatnonzero(ind == 0))
    while rd:
        j = rd.pop(); kd = np.flatnonzero(sub[:, j])
        if kd.size:
            D[kd] = np.maximum(D[kd], D[j]+1); ind[kd] -= 1
            rd.extend(kd[ind[kd] == 0].tolist())
    return int(D.max()) + 1

def intrinsic_pair(R, L, i, j, mid):
    """Две ВНУТРЕННИЕ величины интервала:
       материя  = событий на единицу собственного времени (n / высота^d)
       кривизна = доля прямых связей
    Обе из порядка, координат нет."""
    n = len(mid)
    if n < 12: return None
    h = chain_depth(R, mid)
    if h < 2: return None
    sub = R[np.ix_(mid, mid)]; subL = L[np.ix_(mid, mid)]
    tot = sub.sum()
    if tot == 0: return None
    return n, h, float(n/h**4), float(subL.sum()/tot)
