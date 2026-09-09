import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'waves.py'), encoding='utf-8').read())

def wave_input(pos, directions, freqs, phases0=None):
    """ВОЛНОВОЙ ВХОД. У каждой волны: направление, частота, начальная фаза.
    Шкала события = фаза этой волны в точке события.

    pos        — положения событий, (N, D)
    directions — единичные векторы, (k, D)
    freqs      — частоты, (k,)
    Возвращает (N, k) фаз в [0,1).
    """
    K = np.asarray(directions, dtype=float)
    K = K/np.maximum(np.linalg.norm(K, axis=1, keepdims=True), 1e-12)
    f = np.asarray(freqs, dtype=float)
    p0 = np.zeros(len(f)) if phases0 is None else np.asarray(phases0)
    return np.mod((pos @ K.T)*f[None, :] + p0[None, :], 1.0)

def make_directions(k, D, spread, rs):
    """spread=1 — направления по всей сфере; spread=0 — все параллельны."""
    base = rs.normal(size=(1, D)); base /= np.linalg.norm(base)
    V = rs.normal(size=(k, D))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    W = (1-spread)*base + spread*V
    return W/np.linalg.norm(W, axis=1, keepdims=True)
