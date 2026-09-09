"""Заряд как топологический инвариант.

Идея: заряд — не метка, а свойство формы куска графа: число независимых
циклов в неориентированном графе прямых связей вокруг события.

Порядок ацикличен, но граф связей БЕЗ направления циклы имеет.
"""
import numpy as np


def links_of(R):
    A = np.asarray(R, dtype=np.float32)
    return np.asarray(R & ((A @ A) == 0), dtype=bool)


def neighborhood(R, L, v, radius=1):
    """Окрестность события: оно, его прямые соседи, и так radius раз."""
    N = len(R)
    cur = np.zeros(N, bool); cur[v] = True
    for _ in range(radius):
        nxt = cur | (L[cur].any(0)) | (L[:, cur].any(1))
        cur = nxt
    return np.flatnonzero(cur)


def cycle_number(L, idx):
    """Число независимых циклов в неориентированном подграфе: E - V + C."""
    sub = np.asarray(L[np.ix_(idx, idx)], dtype=bool)
    G = sub | sub.T
    V = len(idx)
    E = int(G.sum()//2)
    # число компонент связности
    seen = np.zeros(V, bool)
    comp = 0
    for s in range(V):
        if seen[s]:
            continue
        comp += 1
        stack = [s]; seen[s] = True
        while stack:
            a = stack.pop()
            for b in np.flatnonzero(G[a]):
                if not seen[b]:
                    seen[b] = True; stack.append(int(b))
    return E - V + comp, V, E, comp
