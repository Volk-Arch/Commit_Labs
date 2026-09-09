import numpy as np
from itertools import permutations, combinations

def random_poset(N, p, rs):
    """Случайный порядок: случайный DAG + транзитивное замыкание."""
    R = np.zeros((N, N), dtype=bool)
    for j in range(1, N):
        R[j, :j] = rs.random(j) < p
    for _ in range(N):
        nxt = R | (R.astype(np.float32) @ R.astype(np.float32) > 0)
        if nxt.sum() == R.sum(): break
        R = nxt
    np.fill_diagonal(R, False)
    return R

def linear_extensions(R):
    """Все линейные продолжения порядка (для малых N)."""
    N = len(R)
    out = []
    def rec(done, rest):
        if not rest:
            out.append(tuple(done)); return
        for x in list(rest):
            if not any(R[x, y] for y in rest if y != x):   # нет предков среди оставшихся
                rec(done+[x], rest-{x})
    rec([], set(range(N)))
    return out

def exact_dimension(R, cap=6):
    """Точная порядковая размерность: min число линейных продолжений,
    пересечение которых даёт R."""
    N = len(R)
    exts = linear_extensions(R)
    if not exts: return None
    # несравнимые пары
    inc = [(i, j) for i in range(N) for j in range(i+1, N)
           if not R[i, j] and not R[j, i]]
    if not inc: return 1
    need = set()
    for i, j in inc:
        need.add((i, j)); need.add((j, i))
    cov = []
    for e in exts:
        pos = {v: t for t, v in enumerate(e)}
        cov.append(frozenset((a, b) for (a, b) in need if pos[a] < pos[b]))
    for k in range(2, cap+1):
        for combo in combinations(range(len(cov)), k):
            u = set()
            for c in combo: u |= cov[c]
            if u == need: return k
    return None
