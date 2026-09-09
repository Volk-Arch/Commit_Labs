import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
from math import gamma
exec(open('/home/claude/rotcones.py').read())

def sprinkle_diamond(N, d, rs):
    """Засев ПРЯМО в интервал Александрова: между (0,0) и (1,0).
    Точка внутри, если t > |x| и (1-t) > |x|."""
    pts = []
    while len(pts) < N:
        m = 8*(N - len(pts)) + 500
        t = rs.uniform(0, 1, m)
        x = rs.uniform(-0.5, 0.5, (m, d-1))
        r = np.linalg.norm(x, axis=1)
        ok = (r < t) & (r < 1-t)
        pts.extend(np.column_stack([t[ok], x[ok]]))
    return np.array(pts[:N])

def rel_true_cone(P):
    dt = P[None, :, 0] - P[:, None, 0]
    dx = np.sqrt(((P[None, :, 1:] - P[:, None, 1:])**2).sum(-1))
    R = (dt > 0) & (dt > dx)
    np.fill_diagonal(R, False)
    return R

def rel_rot_cone(P, nrot):
    d = P.shape[1]
    dt = P[None, :, 0] - P[:, None, 0]
    R = (dt > 0)
    sp = P[:, 1:]
    for a in range(nrot):
        ang = np.pi*a/nrot
        v = np.zeros(d-1); v[0] = np.cos(ang)
        if d > 2: v[1] = np.sin(ang)
        u = sp @ v
        R &= (dt > np.abs(u[None, :] - u[:, None]))
    np.fill_diagonal(R, False)
    return R
