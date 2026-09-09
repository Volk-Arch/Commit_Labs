import numpy as np


def features(R, L, X, g, ng, lev):
    """Всё, что умеем посчитать о группе."""
    Rf = R.astype(np.float32)
    past = Rf[g].sum(1); fut = Rf[:, g].sum(0)
    perm = np.array([tuple(np.argsort(X[v])) for v in g])
    pid = np.array([sum(int(perm[t, i])*4**i for i in range(X.shape[1]))
                    for t in range(len(g))])
    f = {}
    f['размер'] = float(len(g))
    f['окрестность'] = float(len(ng))
    f['глубина'] = float(lev)
    f['заряд'] = float(X[g].var(1).mean())
    f['заряд разброс'] = float(X[g].var(1).std())
    f['время'] = float(X[g].sum(1).mean())
    f['время разброс'] = float(X[g].sum(1).std())
    f['прошлое'] = float(past.mean())
    f['будущее'] = float(fut.mean())
    f['общее прошлое'] = float(R[g].all(axis=0).sum())
    f['прямых внутрь'] = float(L[g].sum())
    f['прямых наружу'] = float(L[:, g].sum())
    f['спин-классов'] = float(len(set(pid.tolist())))
    f['размах фаз'] = float((X[g].max(1)-X[g].min(1)).mean())
    f['компактность'] = float(np.linalg.norm(X[g]-X[g].mean(0), axis=1).mean())
    return f
