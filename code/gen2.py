import numpy as np

def rel_of(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return R

def guided_extension(R, weights, rs):
    """Линейное продолжение, направляемое весами: среди доступных
    выбираем с вероятностью по весу. Это позволяет строить РАЗНЫЕ
    продолжения целенаправленно, а не наугад."""
    Rb = np.ascontiguousarray(np.asarray(R, dtype=bool))
    n = len(Rb)
    indeg = Rb.sum(axis=1).astype(np.int64)
    avail = list(np.flatnonzero(indeg == 0))
    out = []
    while avail:
        w = weights[np.array(avail)]
        w = w - w.min() + 1e-9
        p = w/w.sum()
        t = int(rs.choice(len(avail), p=p))
        v = avail.pop(t)
        out.append(v)
        kids = np.flatnonzero(Rb[:, v])
        if kids.size:
            indeg[kids] -= 1
            for c in kids[indeg[kids] == 0]:
                avail.append(int(c))
    pos = np.empty(n, dtype=np.int64)
    pos[np.array(out)] = np.arange(n)
    return pos

def critical_pairs(R):
    """Несравнимые пары — их и надо 'разводить' продолжениями."""
    Rb = np.asarray(R, dtype=bool)
    inc = (~Rb) & (~Rb.T)
    np.fill_diagonal(inc, False)
    return inc
