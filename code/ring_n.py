import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'triangle.py'), encoding='utf-8').read())

def mix_step_m(Rs, k, rs=None):
    """Кольцо из m миров: вход мира i — сумма от соседа слева, разброс от соседа справа."""
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
