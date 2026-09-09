import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/tarrow.py').read())

def link_graph(R):
    """Граф прямых связей (покрытий) — скелет порядка."""
    Rb = np.ascontiguousarray(R, dtype=bool)
    C = Rb.astype(np.float32); btw = (C.T @ C).astype(np.int32)
    return Rb & (btw.T == 0)

def components(L):
    """Компоненты связности неориентированного скелета."""
    A = L | L.T; n = len(A)
    seen = np.zeros(n, bool); cnt = 0; sizes = []
    for s in range(n):
        if seen[s]: continue
        cnt += 1; stack = [s]; sz = 0
        while stack:
            x = stack.pop()
            if seen[x]: continue
            seen[x] = True; sz += 1
            stack.extend(np.flatnonzero(A[x] & ~seen).tolist())
        sizes.append(sz)
    return cnt, sizes

def cycle_rank(L):
    """Цикломатическое число скелета: E - V + C. Сколько независимых циклов."""
    E = int(L.sum()); V = len(L); C, _ = components(L)
    return E - V + C

def max_antichain_est(R, samples=40):
    """Оценка ширины: наибольший слой по глубине."""
    D = proper_time(np.asarray(R, dtype=np.float32))
    _, cnt = np.unique(D, return_counts=True)
    return int(cnt.max())

def triangles(L):
    """Треугольники в скелете: i->j->k и i->k. В строгом порядке их быть НЕ должно."""
    Lf = L.astype(np.float32)
    return int(np.trace(Lf @ Lf @ Lf))

def global_features(U, R):
    L = link_graph(R)
    C, sizes = components(L)
    n = len(R)
    return {
        'компонент': C,
        'крупнейшая': max(sizes)/n,
        'циклов в скелете': cycle_rank(L),
        'циклов на элемент': cycle_rank(L)/n,
        'ширина (макс. слой)': max_antichain_est(R)/n,
        'высота': int(proper_time(np.asarray(R, dtype=np.float32)).max()),
        'треугольников': triangles(L),
        'связей на элемент': L.sum()/n,
    }
