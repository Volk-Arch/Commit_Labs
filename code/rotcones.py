import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
from math import gamma
exec(open(os.path.join(_HERE, 'cone_rule.py'), encoding='utf-8').read())

def rotated_scales(X, nrot, d):
    """Круглый конус как пересечение ПОВЁРНУТЫХ пирамид.
    Каждый поворот пространственных осей даёт новую пару шкал."""
    N = X.shape[0]
    t = X[:, 0]; sp = X[:, 1:]
    R = np.ones((N, N), dtype=bool)
    dt = t[None, :] - t[:, None]
    R &= (dt > 0)
    for a in range(nrot):
        ang = np.pi*a/nrot
        if d == 2:
            u = sp[:, 0]*np.cos(ang)
            proj = [u]
        else:
            # проецируем на случайное направление, повёрнутое на ang
            v = np.zeros(d-1); v[0] = np.cos(ang); v[1] = np.sin(ang)
            proj = [sp @ v]
        for u in proj:
            du = u[None, :] - u[:, None]
            R &= (dt > np.abs(du))          # конус по этому направлению
    np.fill_diagonal(R, False)
    return R
