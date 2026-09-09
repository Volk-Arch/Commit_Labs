"""Большие миры без хранения матрицы N x N."""
import numpy as np


def sprinkle(N, d, rs, warp=0.0, batch=20000):
    """Засев в конус, float32, малыми пакетами."""
    parts = []; got = 0
    while got < N:
        t = rs.uniform(0, 1, batch).astype(np.float32)
        x = rs.uniform(-0.5, 0.5, (batch, d-1)).astype(np.float32)
        r = np.linalg.norm(x, axis=1)
        ok = (r < t) & (r < 1-t)
        if warp:
            ok &= rs.random(batch) < np.exp(-warp*r)
        if ok.any():
            parts.append(np.column_stack([t[ok], x[ok]]))
            got += int(ok.sum())
        del t, x, r, ok
    P = np.vstack(parts)[:N]
    del parts
    return P


def interval_of(P, i, j):
    """События строго между j (раньше) и i (позже). Без полной матрицы."""
    dt1 = P[:, 0]-P[j, 0]
    m = dt1 > 0
    idx = np.flatnonzero(m)
    if len(idx) == 0: return idx
    d1 = ((P[idx, 1:]-P[j, 1:])**2).sum(1)
    idx = idx[dt1[idx]**2 > d1]
    if len(idx) == 0: return idx
    dt2 = P[i, 0]-P[idx, 0]
    ok = dt2 > 0
    idx = idx[ok]; dt2 = dt2[ok]
    if len(idx) == 0: return idx
    d2 = ((P[idx, 1:]-P[i, 1:])**2).sum(1)
    return idx[dt2**2 > d2]


def height_of(P, idx, cap=6000):
    """Высота интервала. Матрица только для его событий."""
    n = len(idx)
    if n < 2: return 1
    if n > cap: return None
    Q = P[idx]
    dt = Q[None, :, 0]-Q[:, None, 0]
    dx2 = ((Q[None, :, 1:]-Q[:, None, 1:])**2).sum(-1)
    R = (dt > 0) & (dt*dt > dx2)
    del dt, dx2, Q
    ind = R.sum(axis=0).astype(np.int32)
    D = np.zeros(n, dtype=np.int32)
    rd = list(np.flatnonzero(ind == 0))
    while rd:
        a = rd.pop()
        kids = np.flatnonzero(R[a])
        if kids.size:
            D[kids] = np.maximum(D[kids], D[a]+1)
            ind[kids] -= 1
            rd.extend(kids[ind[kids] == 0].tolist())
    h = int(D.max())+1
    del R, ind, D
    return h


def sample_intervals(P, rs, want=60, lo=200, hi=5000, tries=30000):
    N = len(P); out = []
    for _ in range(tries):
        i, j = (int(v) for v in rs.integers(0, N, 2))
        dt = float(P[i, 0]-P[j, 0])
        if dt <= 0: continue
        dx2 = float(((P[i, 1:]-P[j, 1:])**2).sum())
        if dt*dt <= dx2: continue
        mid = interval_of(P, i, j)
        n = len(mid)
        if n < lo or n > hi:
            del mid; continue
        out.append((i, j, mid, float(np.sqrt(dt*dt-dx2))))
        if len(out) >= want: break
    return out
