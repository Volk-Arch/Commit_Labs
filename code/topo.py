import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/conserve.py').read())

def winding_along_cycle(U, cyc):
    """Обмотка вдоль ЗАМКНУТОГО пути в пространстве фаз: суммарный поворот
    проекции на плоскость (0,1). Целочисленна по построению, если путь замкнут."""
    P = U.T.astype(float)
    tot = 0.0; prev = None
    for a in cyc:
        ang = np.arctan2(P[a, 1] - P[cyc[0], 1], P[a, 0] - P[cyc[0], 0])
        if prev is not None:
            d = ang - prev
            while d > np.pi: d -= 2*np.pi
            while d < -np.pi: d += 2*np.pi
            tot += d
        prev = ang
    return tot/(2*np.pi)

def interval_winding(U, R, i, j, ref=None):
    """Обмотка границы интервала: обходим множество событий между i и j
    по порядку углов и считаем, сколько раз обошли центр."""
    Rb = np.ascontiguousarray(R, dtype=bool)
    mid = np.flatnonzero(Rb[:, i] & Rb[j, :])
    if len(mid) < 8: return None, 0
    P = U.T.astype(float)[mid]
    c = P.mean(axis=0)
    ang = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
    order = np.argsort(ang)
    a = ang[order]
    d = np.diff(np.concatenate([a, a[:1]]))
    d = (d + np.pi) % (2*np.pi) - np.pi
    return float(d.sum()/(2*np.pi)), len(mid)
