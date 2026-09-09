import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'spin_try.py'), encoding='utf-8').read())

def loop_one(k, steps=400, radius=0.3, center=None, rs=None):
    """Одна точка обходит замкнутый круг в пространстве фаз.
    Смотрим, что происходит с её псевдоскаляром за полный оборот."""
    r = rs if rs is not None else np.random.default_rng(0)
    c = np.full(k, 0.5) if center is None else np.asarray(center, float)
    # плоскость обхода: два случайных ортогональных направления
    a = r.normal(size=k); a /= np.linalg.norm(a)
    b = r.normal(size=k); b -= (b@a)*a; b /= np.linalg.norm(b)
    th = np.linspace(0, 2*np.pi, steps, endpoint=False)
    P = c[None, :] + radius*(np.cos(th)[:, None]*a[None, :] + np.sin(th)[:, None]*b[None, :])
    return np.mod(P, 1.0)

def sign_track(P):
    """Знак псевдоскаляра вдоль пути и число его смен."""
    s = orientation(P)
    flips = int((s[1:]*s[:-1] < 0).sum())
    closed = int(s[0]*s[-1] < 0)          # сменился ли знак между концом и началом
    return s, flips, closed
