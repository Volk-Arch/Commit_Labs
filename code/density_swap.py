import numpy as np
from collections import deque
rng11 = np.random.default_rng(149)

def grid(k, m):
    cells = np.array(np.unravel_index(np.arange(m**k), (m,)*k)).T
    idx = {tuple(c): i for i, c in enumerate(cells)}
    nbr = [[] for _ in range(len(cells))]
    for i, c in enumerate(cells):
        for d in range(k):
            for s in (-1, 1):
                q = c.copy(); q[d] += s
                if 0 <= q[d] < m: nbr[i].append(idx[tuple(q)])
    return cells, nbr

def swap_with_obstacles(k, m, n_others, rs=None, trials=1):
    """Две копии в кубе фаз, где ЗАНЯТО n_others клеток другими событиями.
    Копия не может встать в занятую клетку (совпали фазы = это одно событие).
    Возможен ли обмен?"""
    r = rs if rs is not None else rng11
    cells, nbr = grid(k, m); n = len(cells)
    ok = 0
    for _ in range(trials):
        free = np.ones(n, bool)
        if n_others > 0:
            occ = r.choice(n, min(n_others, n-2), replace=False)
            free[occ] = False
        fi = np.flatnonzero(free)
        if len(fi) < 2: continue
        a0, b0 = int(fi[0]), int(fi[-1])
        start = (a0, b0); goal = (b0, a0)
        seen = {start}; dq = deque([start]); found = False
        while dq:
            a, b = dq.popleft()
            if (a, b) == goal: found = True; break
            for x in nbr[a]:
                if free[x] and x != b and (x, b) not in seen:
                    seen.add((x, b)); dq.append((x, b))
            for y in nbr[b]:
                if free[y] and y != a and (a, y) not in seen:
                    seen.add((a, y)); dq.append((a, y))
        ok += found
    return ok/max(trials, 1)
