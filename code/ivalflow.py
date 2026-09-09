"""Переносимые величины на интервалах.

Интервал имеет форму. Вопрос: сохраняется ли что-то из формы
при переходе к вложенному или соседнему интервалу.
"""
import numpy as np


def interval_profile(R, mid):
    """Профиль интервала: распределение событий по слоям."""
    n = len(mid)
    sub = np.asarray(R[np.ix_(mid, mid)], dtype=bool)
    dep = np.zeros(n)
    ind = sub.sum(axis=1).astype(np.int64)
    rd = list(np.flatnonzero(ind == 0))
    while rd:
        a = rd.pop(); kd = np.flatnonzero(sub[:, a])
        if kd.size:
            dep[kd] = np.maximum(dep[kd], dep[a]+1); ind[kd] -= 1
            rd.extend(kd[ind[kd] == 0].tolist())
    h = int(dep.max())+1
    prof = np.bincount(dep.astype(int), minlength=h).astype(float)
    return prof/max(prof.sum(), 1), h


def shape_features(R, L, X, i, j, mid):
    """Величины формы интервала."""
    n = len(mid)
    sub = R[np.ix_(mid, mid)]; subL = L[np.ix_(mid, mid)]
    tot = float(sub.sum())
    if tot == 0 or n < 8: return None
    prof, h = interval_profile(R, mid)
    x = np.arange(len(prof))
    cg = float((x*prof).sum()/max(len(prof)-1, 1))
    var = float((x**2*prof).sum()/max((len(prof)-1)**2, 1) - cg**2)
    Xm = X[mid]
    c = Xm - Xm.mean(0)
    sv = np.linalg.svd(c, compute_uv=False)
    sv = sv/max(sv.sum(), 1e-12)
    return dict(n=n, h=h, curv=float(subL.sum()/tot),
                cg=cg, spread=var, aniso=float(sv[0]),
                flat=float(sv[-1]), peak=float(prof.max()))


def nested(R, i, j, mid, rs, want=6):
    """Вложенные интервалы: пары внутри данного."""
    out = []
    if len(mid) < 20: return out
    for _ in range(200):
        a, b = (int(v) for v in rs.choice(mid, 2, replace=False))
        if not R[a, b]: continue
        inner = np.flatnonzero(R[:, b] & R[a, :])
        if len(inner) < 8: continue
        out.append((a, b, inner))
        if len(out) >= want: break
    return out
