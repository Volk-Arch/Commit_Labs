import numpy as np

def rel_of(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return R

def chain_frame(X, chain):
    """Репер, привязанный к ЦЕПИ: направление вдоль цепи + поперечные.
    Это протяжённый объект, у него есть 'начало' и 'конец'."""
    P = X[np.array(chain)]
    d = P[-1] - P[0]
    n = np.linalg.norm(d)
    if n < 1e-12: return None
    e1 = d/n
    # поперечное направление: отклонение середины цепи от прямой
    mid = P[len(P)//2] - (P[0] + 0.5*d)
    m = np.linalg.norm(mid)
    if m < 1e-12: return None
    e2 = mid/m
    e2 = e2 - (e2@e1)*e1
    if np.linalg.norm(e2) < 1e-12: return None
    e2 /= np.linalg.norm(e2)
    return e1, e2

def twist_along(X, chain):
    """Накопленный поворот поперечного направления вдоль цепи —
    аналог скручивания ремня."""
    P = X[np.array(chain)]
    if len(P) < 4: return None
    tot = 0.0; prev = None
    for t in range(1, len(P)-1):
        d = P[t+1] - P[t-1]
        n = np.linalg.norm(d)
        if n < 1e-12: continue
        e1 = d/n
        off = P[t] - 0.5*(P[t+1]+P[t-1])
        off = off - (off@e1)*e1
        m = np.linalg.norm(off)
        if m < 1e-12: continue
        e2 = off/m
        if prev is not None:
            c = np.clip(prev@e2, -1, 1)
            s = np.sign(np.cross(prev[:3], e2[:3])@e1[:3]) if len(e1) >= 3 else 1.0
            tot += s*np.arccos(c)
        prev = e2
    return tot
