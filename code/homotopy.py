import numpy as np
from collections import deque
rng9 = np.random.default_rng(113)

def cube(k, m):
    cells = np.array(np.unravel_index(np.arange(m**k), (m,)*k)).T
    idx = {tuple(c): i for i, c in enumerate(cells)}
    nbr = [[] for _ in range(len(cells))]
    for i, c in enumerate(cells):
        for d in range(k):
            for s in (-1, 1):
                q = c.copy(); q[d] += s
                if 0 <= q[d] < m: nbr[i].append(idx[tuple(q)])
    return cells, nbr

def random_swap_path(k, m, maxsteps=200000, rs=None):
    """Случайное блуждание от (a,b) к (b,a) без совпадения копий."""
    r = rs if rs is not None else rng9
    cells, nbr = cube(k, m)
    n = len(cells); a0, b0 = 0, n - 1
    a, b = a0, b0; traj = [(a, b)]
    for _ in range(maxsteps):
        if r.random() < 0.5:
            cand = [x for x in nbr[a] if x != b]
            if cand: a = int(r.choice(cand))
        else:
            cand = [y for y in nbr[b] if y != a]
            if cand: b = int(r.choice(cand))
        traj.append((a, b))
        if (a, b) == (b0, a0): return cells, traj
    return cells, None

def winding(cells, traj, plane=(0, 1)):
    """Суммарный поворот вектора между копиями в заданной плоскости, в единицах pi."""
    tot = 0.0; prev = None
    for a, b in traj:
        d = cells[b] - cells[a]
        ang = np.arctan2(float(d[plane[1]]), float(d[plane[0]]))
        if prev is not None:
            dd = ang - prev
            while dd > np.pi: dd -= 2*np.pi
            while dd < -np.pi: dd += 2*np.pi
            tot += dd
        prev = ang
    return tot/np.pi
