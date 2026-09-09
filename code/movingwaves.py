import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'wavefield.py'), encoding='utf-8').read())

def moving_phases(pos, t, dirs, freqs, speeds=None, p0=None):
    """ДВИЖУЩИЕСЯ волны: фаза = k·x − ω t. Порядок меняется сам со временем,
    без внешнего правила движения."""
    K = np.asarray(dirs, float)
    K = K/np.maximum(np.linalg.norm(K, axis=1, keepdims=True), 1e-12)
    f = np.asarray(freqs, float)
    v = np.ones_like(f) if speeds is None else np.asarray(speeds, float)
    ph0 = np.zeros_like(f) if p0 is None else np.asarray(p0, float)
    spatial = (pos @ K.T)*f[None, :]
    temporal = (f*v)[None, :]*t
    return np.mod(spatial - temporal + ph0[None, :], 1.0)
