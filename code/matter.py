import numpy as np

def all_maximal_chains(R, L, rs, samples=4000, minlen=4):
    """Максимальные цепи: те, что нельзя продолжить ни вверх, ни вниз."""
    N = len(R)
    Lb = np.asarray(L, dtype=bool)
    starts = np.flatnonzero(Lb.sum(axis=1) == 0)     # без предков
    out = []
    for _ in range(samples):
        if len(starts) == 0: break
        cur = int(rs.choice(starts)); ch = [cur]
        while True:
            nxt = np.flatnonzero(Lb[:, cur])
            if len(nxt) == 0: break
            cur = int(rs.choice(nxt)); ch.append(cur)
        if len(ch) >= minlen: out.append(ch)
    return out

def longest_chain_from(R, L, i, cap=40):
    """Самая длинная цепь из события — жадно по глубине."""
    Lb = np.asarray(L, dtype=bool)
    cur = i; ch = [i]
    for _ in range(cap):
        nxt = np.flatnonzero(Lb[:, cur])
        if len(nxt) == 0: break
        # выбираем потомка с наибольшим будущим
        fut = Lb[:, nxt].sum(axis=0) if nxt.size else None
        cur = int(nxt[int(np.argmax(fut))])
        ch.append(cur)
    return ch
