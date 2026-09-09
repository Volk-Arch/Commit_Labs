import numpy as np

def local_scales(P, R, radius=0.15, minn=25, sample=None, rs=None):
    """Число эффективных шкал ЛОКАЛЬНО: для каждого события берём его
    соседей по координатам и считаем долю связанных пар СРЕДИ НИХ.

    В отличие от группировки по плотности, здесь выборка компактна
    в пространстве, и геометрия отбора не искажает результат."""
    N = len(P)
    idxs = range(N) if sample is None else (rs or np.random.default_rng(0)).choice(N, sample, replace=False)
    out = np.full(N, np.nan)
    for i in idxs:
        r = np.linalg.norm(P - P[i], axis=1)
        nb = np.flatnonzero(r < radius)
        n = len(nb)
        if n < minn: continue
        sub = R[np.ix_(nb, nb)]
        f = sub.sum()/(n*(n-1)/2)
        if 0 < f < 1:
            out[i] = 1 - np.log2(f)
    return out
