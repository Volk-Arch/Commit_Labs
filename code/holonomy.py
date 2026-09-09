"""Зависит ли перенос фазы от пути.

Если два пути между одними событиями дают разную накопленную фазу,
у поля есть напряжённость — из неё строится сила.
"""
import numpy as np


def links_of(R):
    A = np.asarray(R, dtype=np.float32)
    return np.asarray(R & ((A @ A) == 0), dtype=bool)


def paths_between(R, L, a, b, maxlen=6, cap=200):
    """Все пути из a в b по прямым связям (не длиннее maxlen)."""
    out = []
    stack = [(a, [a])]
    while stack and len(out) < cap:
        cur, path = stack.pop()
        if len(path) > maxlen:
            continue
        nxt = np.flatnonzero(L[:, cur])
        for v in nxt:
            if v == b:
                out.append(path + [int(v)])
            elif R[b, v]:
                stack.append((int(v), path + [int(v)]))
    return out


def path_phase(X, path, mode="sum"):
    """Накопленная фаза вдоль пути."""
    if mode == "sum":
        return float(np.sum([X[v].sum() for v in path[1:]]))
    if mode == "step":
        # фаза шага зависит от направления
        s = 0.0
        for i in range(1, len(path)):
            d = X[path[i]] - X[path[i-1]]
            s += float(d[0] - d[-1])
        return s
    raise ValueError(mode)
