import numpy as np
import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from compute import rel_of, links, topo_order

def amplitude_from_phases(X, mode='sum'):
    """Комплексная амплитуда события ИЗ ЕГО ФАЗ, а не приписанная."""
    N, k = X.shape
    if mode == 'sum':
        return np.exp(2j*np.pi*X.sum(1))
    if mode == 'first':
        return np.exp(2j*np.pi*X[:, 0])
    if mode == 'prod':
        return np.prod(np.exp(2j*np.pi*X), axis=1)
    return None

def propagate(L, order, Lb, src_amp, srcs, damp=1.0):
    """Распространение амплитуды: событие суммирует вклады предков,
    умножая на СВОЮ фазу. Никакой нормировки."""
    N = len(Lb)
    A = np.zeros(N, dtype=complex)
    for v in order:
        par = np.flatnonzero(Lb[v])
        if v in srcs:
            A[v] = src_amp[v]
        elif len(par):
            A[v] = damp*src_amp[v]*A[par].sum()/len(par)
    return A
