#!/usr/bin/env python3
"""Общее распространение по правилу «поворот = i» из многих источников (пакеты, щели).
init: {событие: 2-вектор}. Освещены события в будущем любого источника. Монета [[1, iε], [iε, 1]],
по желанию нормированная (делённая на √(1+ε²) — тогда она унитарна)."""
import numpy as np

def propagate(w, LK, R, init, eps=1.0, unit=False, field=None, skip=(), percomp=False, kappa=None, sinks=None):
    I, lab, st, en = LK['I'], LK['lab'], LK['st'], LK['en']
    s = np.ones(len(I), complex) if field is None else field.astype(complex)
    coin = 1j * eps
    if unit: s = s / np.sqrt(1 + eps * eps)
    srcs = np.array(sorted(init)); lit_ev = R[srcs].any(0); lit_ev[srcs] = True
    lit = lit_ev[I]
    psi = np.zeros((w.n, 2), complex)
    for k, v in init.items(): psi[k] = v
    first = srcs.min(); skip = set(skip)
    for j in range(first + 1, w.n):
        if j in init or j in skip: continue
        a, b = st[j], en[j]
        if a >= b: continue
        li = lit[a:b] if sinks is None else lit[a:b] & ~sinks[I[a:b]]   # поглотители не излучают дальше
        nl = li.sum()
        if nl == 0: continue
        ii, mm, ss = I[a:b][li], lab[a:b][li], s[a:b][li]
        c0 = ss * (psi[ii, 0] + coin * psi[ii, 1]); c1 = ss * (psi[ii, 1] + coin * psi[ii, 0])
        if percomp:   # каждая компонента — среднее по тем звеньям, что в неё приходят
            n0, n1 = (mm != 1).sum(), (mm != 0).sum()
            psi[j, 0] = (c0[mm == 0].sum() + c0[mm == 2].sum()) / max(n0, 1)
            psi[j, 1] = (c1[mm == 1].sum() + c1[mm == 2].sum()) / max(n1, 1)
        elif kappa is not None:   # сумма историй: одно число κ на весь мир вместо среднего
            psi[j, 0] = (c0[mm == 0].sum() + c0[mm == 2].sum()) * kappa
            psi[j, 1] = (c1[mm == 1].sum() + c1[mm == 2].sum()) * kappa
        else:
            psi[j, 0] = (c0[mm == 0].sum() + c0[mm == 2].sum()) / nl
            psi[j, 1] = (c1[mm == 1].sum() + c1[mm == 2].sum()) / nl
    return psi
