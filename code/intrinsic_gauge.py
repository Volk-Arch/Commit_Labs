"""Фаза на связи, вычисляемая ИЗ ПОРЯДКА, а не из координат.

Чтобы калибровочное поле было выведено, а не вписано, величина на связи
должна следовать из структуры причинного множества.
"""
import numpy as np


def link_quantities(R, L):
    """Набор величин на каждой прямой связи, вычисляемых из порядка."""
    N = len(R)
    Rf = np.asarray(R, dtype=np.float32)
    past = Rf.sum(1)          # сколько предков
    future = Rf.sum(0)        # сколько потомков
    common_past = Rf @ Rf.T   # общие предки пары
    common_fut = Rf.T @ Rf    # общие потомки
    out = {}
    ii, jj = np.nonzero(L)    # L[i,j]: j раньше i
    out['idx'] = (ii, jj)
    # антисимметричные кандидаты (меняют знак при обмене концов)
    out['разность прошлого'] = past[ii] - past[jj]
    out['разность будущего'] = future[ii] - future[jj]
    out['перекос конусов'] = (past[ii]-future[ii]) - (past[jj]-future[jj])
    # симметричные (для контроля)
    out['общие предки'] = common_past[ii, jj]
    out['общие потомки'] = common_fut[ii, jj]
    return out


def holonomy(R, L, phase_map, rs, maxlen=5, want=60, tries=40000):
    """Разброс накопленной фазы между путями ОДИНАКОВОЙ длины."""
    N = len(R)
    res = []
    for _ in range(tries):
        a, b = (int(v) for v in rs.integers(0, N, 2))
        if not R[b, a]:
            continue
        paths = _paths(R, L, a, b, maxlen)
        byl = {}
        for p in paths:
            byl.setdefault(len(p), []).append(p)
        for ln, g in byl.items():
            if len(g) < 2:
                continue
            ph = [sum(phase_map.get((p[i], p[i-1]), 0.0)
                      for i in range(1, len(p))) for p in g]
            res.append(float(np.std(ph)))
        if len(res) >= want:
            break
    return np.array(res)


def _paths(R, L, a, b, maxlen, cap=60):
    out = []
    stack = [(a, [a])]
    while stack and len(out) < cap:
        cur, path = stack.pop()
        if len(path) > maxlen:
            continue
        for v in np.flatnonzero(L[:, cur]):
            v = int(v)
            if v == b:
                out.append(path + [v])
            elif R[b, v]:
                stack.append((v, path + [v]))
    return out
