import numpy as np

def pt(K):
    Kb = np.ascontiguousarray(K, dtype=bool); n = len(Kb)
    ind = Kb.sum(axis=1).astype(np.int64); D = np.zeros(n, dtype=np.int64)
    rd = list(np.flatnonzero(ind == 0))
    while rd:
        j = rd.pop(); kd = np.flatnonzero(Kb[:, j])
        if kd.size:
            D[kd] = np.maximum(D[kd], D[j]+1); ind[kd] -= 1
            rd.extend(kd[ind[kd] == 0].tolist())
    return D

def grow_alt(N, rs, phase=1.0, nlinks=2, horizon=6, burn=0.0, every=1):
    """Рост с квантовым шагом и ВЫРАВНИВАНИЕМ:
    burn — доля связей, вычитаемых там, где амплитуда погасла."""
    R = np.zeros((N, N), dtype=bool); D = np.zeros(N, dtype=np.int64)
    for n in range(1, N):
        if n <= nlinks:
            past = np.ones(n, dtype=bool)
        else:
            A = R[:n, :n].astype(np.complex128)*np.exp(1j*phase)
            seed = np.ones(n, dtype=np.complex128)*(D[:n] == 0)
            if seed.sum() == 0: seed = np.ones(n, dtype=np.complex128)
            acc = seed.copy(); cur = seed.copy()
            for _ in range(horizon):
                cur = A.T @ cur; acc = acc + cur
                if np.abs(cur).max() < 1e-14: break
            p = np.abs(acc)**2
            if p.sum() <= 0 or np.count_nonzero(p) < nlinks: p = np.ones(n)
            p = p/p.sum()
            sel = rs.choice(n, min(nlinks, n), replace=False, p=p)
            past = np.zeros(n, dtype=bool); past[sel] = True
        idx = np.flatnonzero(past)
        R[:n, n] = past | R[:n, :n][:, idx].any(axis=1)
        D[n] = D[:n][R[:n, n]].max()+1 if R[:n, n].any() else 0
        # ВЫРАВНИВАНИЕ: убираем связи у самых 'горячих' событий
        if burn > 0 and n % every == 0 and n > 20:
            deg = R[:n+1, :n+1].sum(axis=1)
            hot = np.argsort(-deg)[:max(1, int(burn*n))]
            for h in hot:
                cs = np.flatnonzero(R[h, :n+1])
                if len(cs) > 2:
                    drop = rs.choice(cs, max(1, len(cs)//4), replace=False)
                    R[h, drop] = False
    return R
