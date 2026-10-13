import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/generated.py').read())

def order_to_input(R, k, rs=None):
    """Из порядка достаём k величин для СЛЕДУЮЩЕГО мира.
    Берём то, что порядок реально помнит: сумму (глубина) и разброс,
    плюс развёртка через общую форму — как в упрощённом входе."""
    r = rs if rs is not None else np.random.default_rng(0)
    Rb = np.ascontiguousarray(np.asarray(R, dtype=bool))
    Rf = Rb.astype(np.float32); N = len(Rb)
    dep = proper_time(Rf).astype(float)
    codep = proper_time(np.ascontiguousarray(Rf.T)).astype(float)
    S = dep - codep                              # «время» события
    past = Rf.sum(axis=1); fut = Rf.sum(axis=0)
    V = np.abs(past - fut)                       # «разброс»
    S = np.argsort(np.argsort(S + 1e-9*r.random(N)))/N
    V = np.argsort(np.argsort(V + 1e-9*r.random(N)))/N
    shape = np.array([np.sin(2*np.pi*m/k + 0.7) for m in range(k)])
    shape -= shape.mean(); shape /= max(shape.std(), 1e-9)
    return np.mod(S[:, None] + 5.0*V[:, None]*shape[None, :], 1.0)
