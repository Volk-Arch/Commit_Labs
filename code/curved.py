import numpy as np
import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from cliff import clifford, transposition_op
from spindyn import rel_of, proper_time

def rotor(g, i, j, theta):
    B = g[i] @ g[j]
    return np.cos(theta/2)*np.eye(len(B), dtype=complex) + np.sin(theta/2)*B

def transport_weighted(X, chain, g, k, alpha=0.0):
    """Перенос с ДОБАВКОЙ, зависящей от того, насколько близко подошли фазы.
    alpha=0 — как раньше (плоско). alpha>0 — добавляем поворот по величине сближения."""
    d = g[0].shape[0]
    M = np.eye(d, dtype=complex)
    perms = [tuple(np.argsort(X[c])) for c in chain]
    for t in range(1, len(perms)):
        a, b = perms[t-1], perms[t]
        pa = list(a)
        for i in range(k):
            if pa[i] != b[i]:
                j = pa.index(b[i])
                M = M @ transposition_op(g, i, j)
                if alpha:
                    # добавка: поворот по величине разности фаз в момент смены
                    gap = abs(X[chain[t], pa[i]] - X[chain[t], pa[j]])
                    M = M @ rotor(g, pa[i], pa[j], alpha*gap)
                pa[i], pa[j] = pa[j], pa[i]
    return M
