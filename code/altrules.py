import numpy as np

def acyclic(R):
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

def transitive_frac(R, samples=60000, rs=None):
    r = rs if rs is not None else np.random.default_rng(0)
    Rb = np.ascontiguousarray(R, dtype=bool); n = len(Rb)
    ok = tot = 0
    for _ in range(samples):
        a, b, c = (int(v) for v in r.integers(0, n, 3))
        if Rb[b, a] and Rb[c, b]:
            tot += 1; ok += bool(Rb[c, a])
    return ok/max(tot, 1)

def rules(X):
    """Разные правила построения отношения из тех же k чисел."""
    N, k = X.shape
    D = X[None, :, :] - X[:, None, :]      # D[i,j,m] = x_j[m] - x_i[m]
    out = {}
    out['все больше (наше)']      = np.all(D > 0, axis=2)
    out['сумма больше']           = D.sum(axis=2) > 0
    out['произведение положит.']  = np.all(D > 0, axis=2) | (np.prod(D, axis=2) > 0) & (D.sum(axis=2) > 0)
    out['максимум < 0 запрещён']  = (D.min(axis=2) > 0)
    out['большинство больше']     = (D > 0).sum(axis=2) > k/2
    out['все больше на delta']    = np.all(D > 0.05, axis=2)
    out['первая больше и сумма']  = (D[:, :, 0] > 0) & (D.sum(axis=2) > 0)
    out['евклидов конус']         = (D[:, :, 0] > 0) & (D[:, :, 0]**2 > (D[:, :, 1:]**2).sum(-1))
    for k_ in out: np.fill_diagonal(out[k_], False)
    return out
