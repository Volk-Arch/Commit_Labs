"""Классический последовательный рост Райдаута–Соркина.

События только РОЖДАЮТСЯ, по одному. Новый элемент выбирает себе
прошлое — «частичный стебель» (замкнутое вниз подмножество).

Общее решение задаётся набором констант связи t_n: вероятность того,
что новый элемент получит прошлое размера n, пропорциональна
числу таких подмножеств, умноженному на t_n.
"""
import numpy as np


def grow_csg(N, t, rs):
    """t — вектор констант связи t_0..t_{N-1}.
    t = [1,0,0,...] даёт цепь; t всё единицы — антицепь;
    геометрическая t даёт транзитивную перколяцию."""
    R = np.zeros((N, N), dtype=bool)
    for n in range(1, N):
        # каждый предыдущий элемент выбирается независимо с вероятностью p
        # (это случай транзитивной перколяции — простейшая модель CSG)
        anc = rs.random(n) < t
        if anc.any():
            idx = np.flatnonzero(anc)
            R[:n, n] = anc | R[:n, :n][:, idx].any(axis=1)
    return R


def grow_csg_general(N, tvec, rs, cap=60):
    """Общий CSG: вероятность прошлого размера m ~ C(n,m)*t_m.
    Выбираем размер прошлого, затем случайный стебель этого размера."""
    R = np.zeros((N, N), dtype=bool)
    for n in range(1, N):
        mmax = min(n, len(tvec)-1, cap)
        w = np.array([tvec[m]*float(np.math.comb(n, m)) if m <= n else 0.0
                      for m in range(mmax+1)], dtype=float)
        if w.sum() <= 0:
            m = 0
        else:
            m = int(rs.choice(len(w), p=w/w.sum()))
        if m > 0:
            sel = rs.choice(n, m, replace=False)
            R[:n, n] = R[:n, :n][:, sel].any(axis=1)
            R[sel, n] = True
    return R
