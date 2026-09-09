import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'cascade.py'), encoding='utf-8').read())

def ring_step(Rs, k, rs=None):
    """Кольцо миров: вход мира i строится из выхода мира i-1."""
    r = rs if rs is not None else np.random.default_rng(0)
    m = len(Rs)
    new = []
    for i in range(m):
        src = Rs[(i-1) % m]
        X = order_to_input(src, k, np.random.default_rng(int(r.integers(0, 10**9))))
        new.append(rel_of(X))
    return new

def mix_step(Rs, k, rs=None):
    """Вход мира i строится из ДВУХ соседей: берём сумму от одного, разброс от другого."""
    r = rs if rs is not None else np.random.default_rng(0)
    m = len(Rs); new = []
    for i in range(m):
        A = np.asarray(Rs[(i-1) % m], dtype=np.float32)
        B = np.asarray(Rs[(i+1) % m], dtype=np.float32)
        N = len(A)
        S = proper_time(A).astype(float) - proper_time(np.ascontiguousarray(A.T)).astype(float)
        V = np.abs(B.sum(axis=1) - B.sum(axis=0))
        S = np.argsort(np.argsort(S + 1e-9*r.random(N)))/N
        V = np.argsort(np.argsort(V + 1e-9*r.random(N)))/N
        shape = np.array([np.sin(2*np.pi*j/k + 0.7) for j in range(k)])
        shape -= shape.mean(); shape /= max(shape.std(), 1e-9)
        new.append(rel_of(np.mod(S[:, None] + 5.0*V[:, None]*shape[None, :], 1.0)))
    return new
