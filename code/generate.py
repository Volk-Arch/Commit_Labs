import numpy as np

def rel_of(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return R

def random_linear_extension(R, rs):
    """Случайное линейное продолжение порядка: топологическая сортировка
    со случайным выбором среди доступных."""
    Rb = np.ascontiguousarray(np.asarray(R, dtype=bool))
    n = len(Rb)
    indeg = Rb.sum(axis=1).astype(np.int64)
    avail = list(np.flatnonzero(indeg == 0))
    out = []
    while avail:
        t = int(rs.integers(0, len(avail)))
        v = avail.pop(t)
        out.append(v)
        kids = np.flatnonzero(Rb[:, v])
        if kids.size:
            indeg[kids] -= 1
            for c in kids[indeg[kids] == 0]:
                avail.append(int(c))
    return np.array(out)

def scales_from_extensions(R, k, rs, tries=200):
    """Порождаем k шкал, пересечение которых даёт R (или близкое к нему)."""
    n = len(R)
    Rb = np.asarray(R, dtype=bool)
    exts = []
    for _ in range(tries):
        e = random_linear_extension(R, rs)
        pos = np.empty(n, dtype=np.int64)
        pos[e] = np.arange(n)
        exts.append(pos)
        if len(exts) >= k*8: break
    # жадно набираем k штук, чтобы пересечение было ближе к R
    chosen = []
    cur = np.ones((n, n), dtype=bool); np.fill_diagonal(cur, False)
    for _ in range(k):
        best, bi = None, -1
        for t, p in enumerate(exts):
            if t in chosen: continue
            cand = cur & (p[None, :] > p[:, None])
            extra = int((cand & ~Rb).sum())
            if best is None or extra < best:
                best, bi = extra, t
        if bi < 0: break
        chosen.append(bi)
        cur = cur & (exts[bi][None, :] > exts[bi][:, None])
        np.fill_diagonal(cur, False)
    return np.column_stack([exts[t] for t in chosen])/n, cur
