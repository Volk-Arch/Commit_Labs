import numpy as np
from itertools import combinations

def sprinkle(N, d, rs, T=1.0):
    pts = []
    while len(pts) < N:
        m = 8*(N-len(pts))+200
        t = rs.uniform(0, T, m); x = rs.uniform(-T/2, T/2, (m, d-1))
        r = np.linalg.norm(x, axis=1); ok = (r < t) & (r < T-t)
        pts.extend(np.column_stack([t[ok], x[ok]]))
    P = np.array(pts[:N]); return P[np.argsort(P[:, 0])]

def order_of(P):
    t = P[:, 0]; x = P[:, 1:]
    dt = t[None, :]-t[:, None]
    dx2 = ((x[None, :, :]-x[:, None, :])**2).sum(-1)
    R = (dt > 0) & (dt**2 > dx2)
    return R.T          # R[i,j]: j предшествует i  -> приводим к R[a,b]=True если b<a

def dim_upper(R, tries=4000, cap=12, rs=None):
    """Верхняя оценка порядковой размерности: жадное покрытие
    случайными линейными продолжениями."""
    r = rs if rs is not None else np.random.default_rng(0)
    N = len(R)
    Rb = np.asarray(R, dtype=bool)
    inc = [(i, j) for i in range(N) for j in range(i+1, N)
           if not Rb[i, j] and not Rb[j, i]]
    if not inc: return 1
    need = set()
    for i, j in inc: need.add((i, j)); need.add((j, i))
    # генерируем случайные линейные продолжения
    exts = []
    for _ in range(tries):
        avail = set(range(N)); done = []
        ok = True
        while avail:
            free = [x for x in avail if not any(Rb[x, y] for y in avail if y != x)]
            if not free: ok = False; break
            done.append(int(r.choice(free))); avail.discard(done[-1])
        if ok: exts.append(done)
    covs = []
    for e in exts:
        pos = {v: t for t, v in enumerate(e)}
        covs.append(frozenset((a, b) for (a, b) in need if pos[a] < pos[b]))
    rem = set(need); used = 0
    while rem and used < cap:
        best = max(covs, key=lambda c: len(c & rem))
        g = len(best & rem)
        if g == 0: return None
        rem -= best; used += 1
    return used if not rem else None
