import numpy as np

def worldlines(R, L, rs, count=200, minlen=4, maxlen=30):
    """Мировые линии: цепи по прямым связям."""
    N = len(R)
    Lb = np.asarray(L, dtype=bool)
    out = []
    for _ in range(count*20):
        s = int(rs.integers(0, N))
        cur = s; ch = [s]
        for _ in range(maxlen):
            nxt = np.flatnonzero(Lb[:, cur])
            if len(nxt) == 0: break
            cur = int(rs.choice(nxt)); ch.append(cur)
        if len(ch) >= minlen: out.append(ch)
        if len(out) >= count: break
    return out

def line_density(N, lines):
    """Псевдоматерия: сколько мировых линий проходит через событие."""
    d = np.zeros(N)
    for ch in lines:
        for v in ch: d[v] += 1
    return d

def local_stats(R, L, idx):
    """Структурные величины на подмножестве."""
    n = len(idx)
    if n < 20: return None
    sub = R[np.ix_(idx, idx)]; subL = L[np.ix_(idx, idx)]
    tot = float(sub.sum())
    if tot == 0: return None
    return float(subL.sum()/tot), float(tot/(n*(n-1)/2))
