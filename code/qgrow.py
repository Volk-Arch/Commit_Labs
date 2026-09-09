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

def grow_local(N, rs, phase=0.0, frac=0.05, horizon=3, base=0.05):
    """Амплитуда считается ЛОКАЛЬНО: пути длины <= horizon от недавнего фронта.
    Пути короткие и многочисленные -> амплитуды реально складываются."""
    R = np.zeros((N, N), dtype=bool)
    D = np.zeros(N, dtype=np.int64)
    for n in range(1, N):
        if n < 10:
            past = rs.random(n) < 0.4
        else:
            A = R[:n, :n]
            Dn = D[:n]
            top = Dn.max()
            # фронт: события последних horizon слоёв
            front = (Dn >= top-horizon)
            amp = np.zeros(n, dtype=np.complex128)
            amp[front] = 1.0
            W = A.astype(np.complex128)*np.exp(1j*phase)
            acc = amp.copy(); cur = amp.copy()
            for _ in range(horizon):
                cur = W.T @ cur        # распространяем ВПЕРЁД по связям
                acc = acc + cur
                if np.abs(cur).max() < 1e-14: break
            p = np.abs(acc)**2 + base   # base не даёт занулиться
            p = p/p.sum()
            m = max(1, int(frac*n))
            sel = rs.choice(n, min(m, n), replace=False, p=p)
            past = np.zeros(n, dtype=bool); past[sel] = True
        if past.any():
            idx = np.flatnonzero(past)
            R[:n, n] = past | R[:n, :n][:, idx].any(axis=1)
            D[n] = D[:n][R[:n, n]].max()+1 if R[:n, n].any() else 0
    return R
