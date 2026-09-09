import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'charge.py'), encoding='utf-8').read())

def charge_weights(X, R, g=1.0, base=1.0):
    """Поле действует на ВЕС связи, а не на фазы. Порядок не трогаем вовсе.
    Одинаковые метки — связь сильнее, разные — слабее."""
    lab, _ = labels_of(X)
    Rb = np.asarray(R, dtype=bool)
    same = (lab[None, :] == lab[:, None]) & (lab[None, :] >= 0) & (lab[:, None] >= 0)
    diff = (lab[None, :] != lab[:, None]) & (lab[None, :] >= 0) & (lab[:, None] >= 0)
    W = np.where(Rb, base, 0.0)
    W[Rb & same] = base*(1+g)
    W[Rb & diff] = base/(1+g)
    return W, lab

def field_of(W, R, src, alpha=0.02):
    """Поле от источника по ВЗВЕШЕННЫМ связям."""
    from scipy.linalg import solve_triangular
    Wf = np.ascontiguousarray(W, dtype=np.float32); n = len(Wf)
    M = np.eye(n, dtype=np.float32) + alpha*Wf
    return 0.5*solve_triangular(M, Wf[:, src].copy(), lower=True, unit_diagonal=True)
