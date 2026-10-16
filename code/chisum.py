#!/usr/bin/env python3
"""Сумма историй С КИРАЛЬНОСТЬЮ: ψ_m(K) = κ·Σ (ψ_m + iε ψ_{1−m}) по звеньям, κ — одно на мир. Веса вещественны,
значит теорема чётности (и точная унитарность делителей по зеркалу) сохраняется. Ищем κ_c и проверяем норму."""
import sys, math, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
from prop import propagate
sys.argv = _a
w = CH.sym_world(4.0, 24.0); R = CH.build_order(w, 'proj'); LK = CH.prep(w, CH.links_of(R), 1); X = w.X
band0 = np.nonzero(np.abs(X[:, 0] + 10) < 0.35)[0]; times = np.arange(-8, 9, 2.0)
def sl(psi, f):
    return np.array([f(psi[(np.abs(X[:, 0] - t) < 0.35) & (np.abs(X[:, 1]) < 20 - abs(t))]) for t in times])
res = {}
for eps in (0.25, 0.5, 1.0):
    lo, hi = 0.01, 1.0
    for _ in range(30):
        k = math.sqrt(lo * hi)
        v = sl(propagate(w, LK, R, {b: np.array([1, 0], complex) for b in band0}, eps=eps, unit=True, kappa=k), lambda p: np.sqrt((np.abs(p) ** 2).sum(1)).mean())
        if np.polyfit(times[2:], np.log(v[2:] + 1e-300), 1)[0] > 0: hi = k
        else: lo = k
    res[eps] = k
    v = sl(propagate(w, LK, R, {b: np.array([1, 0], complex) for b in band0}, eps=eps, unit=True, kappa=k), lambda p: np.sqrt((np.abs(p) ** 2).sum(1)).mean())
    init = {b: np.array([math.exp(-X[b, 1] ** 2 / 8), 0], complex) for b in band0 if abs(X[b, 1]) < 7}
    p2 = sl(propagate(w, LK, R, init, eps=eps, unit=True, kappa=k), lambda p: (np.abs(p) ** 2).sum())
    print(f'ε = {eps}: κ_c = {k:.4f}; плоская волна |ψ|: ' + ' '.join(f'{x:.2f}' for x in v / v[0]) + '; пакет Σ|ψ|²: ' + ' '.join(f'{x:.2f}' for x in p2 / p2[0]))
np.save('/home/claude/data/chisum_kc.npy', res, allow_pickle=True)
