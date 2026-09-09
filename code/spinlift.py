import numpy as np
import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from cliff import clifford, transposition_op

def perm_to_transpositions(p):
    """Разложение перестановки в транспозиции соседних."""
    p = list(p); out = []
    for i in range(len(p)):
        for j in range(len(p)-1):
            if p[j] > p[j+1]:
                p[j], p[j+1] = p[j+1], p[j]; out.append(j)
    return out

def lift(p, gens):
    """Подъём перестановки в двойное накрытие: произведение операторов."""
    k = len(gens)
    M = np.eye(gens[0].shape[0], dtype=complex)
    for j in perm_to_transpositions(p):
        M = M @ transposition_op(gens, j, j+1)
    return M

def event_spinor(X, gens, v0=None):
    """Спинор события: подъём его ранжировки шкал, применённый к опорному вектору."""
    d = gens[0].shape[0]
    if v0 is None:
        v0 = np.zeros(d, dtype=complex); v0[0] = 1.0
    out = np.zeros((len(X), d), dtype=complex)
    for t, row in enumerate(X):
        p = tuple(np.argsort(row))
        out[t] = lift(p, gens) @ v0
    return out
