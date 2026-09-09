import numpy as np

def longest_chain(Lb, a, b, maxlen=25):
    """Геодезическая: самая ДЛИННАЯ цепь от a до b по прямым связям.
    В каузальных множествах максимальная цепь = геодезическая."""
    n = len(Lb)
    best = {a: [a]}
    frontier = [a]
    for _ in range(maxlen):
        nxt = []
        for v in frontier:
            for w in np.flatnonzero(Lb[:, v]):
                w = int(w)
                cand = best[v] + [w]
                if w not in best or len(cand) > len(best[w]):
                    best[w] = cand
                    nxt.append(w)
        if not nxt: break
        frontier = list(set(nxt))
        if b in best and len(frontier) > 4000: break
    return best.get(b)

def geodesics(Lb, R, X, centre, rs, want=200, minlen=5):
    """Набираем геодезические и смотрим, как меняется радиус вдоль них."""
    N = len(Lb)
    out = []
    for _ in range(4000):
        i, j = (int(v) for v in rs.integers(0, N, 2))
        if not R[i, j]: continue
        ch = longest_chain(Lb, j, i)
        if ch is None or len(ch) < minlen: continue
        r = np.linalg.norm(X[ch] - centre, axis=1)
        out.append((len(ch), float(r[0]), float(r[-1]), float(r.min())))
        if len(out) >= want: break
    return np.array(out) if out else None
