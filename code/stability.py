import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'intrinsic.py'), encoding='utf-8').read())

def k_order_pts(N, k, rs):
    U = np.array([rs.permutation(N) for _ in range(k)], dtype=np.int64)
    R = np.ones((N, N), dtype=bool)
    for u in U: R &= (u[None, :] < u[:, None])
    return U, R

def can_exchange(k, m=6):
    """Возможен ли обмен двух копий в k-мерном кубе фаз."""
    from collections import deque
    cells = np.array(np.unravel_index(np.arange(m**k), (m,)*k)).T
    idx = {tuple(c): i for i, c in enumerate(cells)}
    nbr = [[] for _ in range(len(cells))]
    for i, c in enumerate(cells):
        for d in range(k):
            for s in (-1, 1):
                q = c.copy(); q[d] += s
                if 0 <= q[d] < m: nbr[i].append(idx[tuple(q)])
    n = len(cells); a0, b0 = 0, n-1
    seen = {(a0, b0)}; dq = deque([(a0, b0)])
    while dq:
        a, b = dq.popleft()
        if (a, b) == (b0, a0): return True
        for x in nbr[a]:
            if x != b and (x, b) not in seen: seen.add((x, b)); dq.append((x, b))
        for y in nbr[b]:
            if y != a and (a, y) not in seen: seen.add((a, y)); dq.append((a, y))
    return False

def chirality_possible(U, R):
    """Есть ли псевдоскаляр: нужен определитель, то есть k>=3."""
    k = U.shape[0]
    if k < 3: return False, 0.0
    d, ok = directions(U, R)
    P = U.T.astype(np.float64); N = len(P)
    D = P[None, :, :3] - P[:, None, :3]
    cr = np.cross(np.broadcast_to(d[:, None, :3], (N, N, 3)),
                  np.broadcast_to(d[None, :, :3], (N, N, 3)))
    det = (cr*D).sum(-1)
    W = R.astype(np.float64); W[R & (det.T < 0)] = 0.001
    V = U.copy(); V[[0, 1]] = V[[1, 0]]
    Rv = np.ones((N, N), dtype=bool)
    for u in V: Rv &= (u[None, :] < u[:, None])
    dv, _ = directions(V, Rv)
    Pv = V.T.astype(np.float64); Dv = Pv[None, :, :3] - Pv[:, None, :3]
    crv = np.cross(np.broadcast_to(dv[:, None, :3], (N, N, 3)),
                   np.broadcast_to(dv[None, :, :3], (N, N, 3)))
    detv = (crv*Dv).sum(-1)
    Wv = Rv.astype(np.float64); Wv[Rv & (detv.T < 0)] = 0.001
    return True, float(np.abs(W - Wv).sum())
