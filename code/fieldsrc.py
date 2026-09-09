"""Напряжённость калибровочного поля и её источник.

Фаза на связи: phi(i->j) = общие предки - общие потомки.
Напряжённость (кривизна): разброс суммы фаз между путями с общими концами.
"""
import numpy as np


def link_phase_matrix(R):
    """Матрица фазы на связях, вычисленная из порядка."""
    Rf = np.asarray(R, dtype=np.float32)
    cp = Rf @ Rf.T      # общие предки
    cf = Rf.T @ Rf      # общие потомки
    return cp - cf


def paths_between(R, L, a, b, maxlen=4, cap=40):
    out = []
    st = [(a, [a])]
    while st and len(out) < cap:
        cur, p = st.pop()
        if len(p) > maxlen:
            continue
        for v in np.flatnonzero(L[:, cur]):
            v = int(v)
            if v == b:
                out.append(p + [v])
            elif R[b, v]:
                st.append((v, p + [v]))
    return out


def curvature_at(R, L, PH, a, b, maxlen=4):
    """Кривизна для пары: разброс фазы между путями одной длины."""
    ps = paths_between(R, L, a, b, maxlen)
    byl = {}
    for p in ps:
        byl.setdefault(len(p), []).append(p)
    vals = []
    for ln, g in byl.items():
        if len(g) < 2:
            continue
        s = [sum(float(PH[p[i], p[i-1]]) for i in range(1, len(p))) for p in g]
        vals.append(float(np.std(s)))
    return float(np.mean(vals)) if vals else np.nan
