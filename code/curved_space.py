import numpy as np

def sprinkle_flat(N, d, rs):
    """Равномерный засев в плоский конус — кривизна ноль."""
    P = []
    while len(P) < N:
        m = 8*(N-len(P))+200
        t = rs.uniform(0, 1, m); x = rs.uniform(-0.5, 0.5, (m, d-1))
        r = np.linalg.norm(x, axis=1); ok = (r < t) & (r < 1-t)
        P.extend(np.column_stack([t[ok], x[ok]]))
    return np.array(P[:N])

def sprinkle_curved(N, d, rs, warp=0.0):
    """Засев с ПЕРЕМЕННОЙ плотностью — аналог кривизны.
    warp > 0: гуще к центру (как около массы)."""
    P = []
    while len(P) < N:
        m = 12*(N-len(P))+300
        t = rs.uniform(0, 1, m); x = rs.uniform(-0.5, 0.5, (m, d-1))
        r = np.linalg.norm(x, axis=1)
        ok = (r < t) & (r < 1-t)
        t, x, r = t[ok], x[ok], r[ok]
        if warp != 0 and len(t):
            # принимаем с вероятностью, растущей к центру
            keep = rs.random(len(t)) < np.exp(-warp*r)
            t, x = t[keep], x[keep]
        if len(t): P.extend(np.column_stack([t, x]))
    return np.array(P[:N])

def order_of(P):
    dt = P[None, :, 0]-P[:, None, 0]
    dx2 = ((P[None, :, 1:]-P[:, None, 1:])**2).sum(-1)
    return np.asarray(((dt > 0) & (dt**2 > dx2)).T, dtype=bool)
