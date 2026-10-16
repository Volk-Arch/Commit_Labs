#!/usr/bin/env python3
"""ИЩЕМ СОХРАНЯЕМУЮ НОРМУ правила «поворот = i» (монета нормирована, унитарна).

Событие — среднее по освещённым входящим звеньям: ψ_K = (1/k) Σ U_JK ψ_J, |U ψ| = |ψ|.
Отсюда (неравенство треугольника) |ψ_K| ≤ max|ψ_J| и |ψ_K|² ≤ (1/k)Σ|ψ_J|²: точечная норма не растёт
и сохраняется ровно тогда, когда все приходящие истории согласны (когерентны).

Проверки:
  A. «Плоская волна» — источник на всём срезе t ≈ −10 с одинаковым вектором: как меняется |ψ| во времени.
  B. Пакет (гаусс шириной 2 по x) — суммы по срезам: N₁ = Σ|ψ|², N_in = Σ k_in|ψ|², N_out = Σ k_out|ψ|²,
     и «линейный поток» Σ ψ-вес (для ε = 0). Что из этого постоянно?
"""
import sys, math, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
from prop import propagate
sys.argv = _a

w = CH.sym_world(4.0, 24.0); R = CH.build_order(w, 'proj'); Lk = CH.links_of(R); LK = CH.prep(w, Lk, 1); X = w.X
kin, kout = Lk.sum(0), Lk.sum(1)
band0 = np.nonzero(np.abs(X[:, 0] + 10) < 0.35)[0]
times = np.arange(-8, 9, 2.0)
def slices(psi, m=None):
    out = []
    for t in times:
        s = np.nonzero((np.abs(X[:, 0] - t) < 0.35) & (np.abs(X[:, 1]) < 24 - abs(t) - 1))[0]
        a2 = (np.abs(psi[s]) ** 2).sum(1)
        out.append((a2.sum(), (a2 * kin[s]).sum(), (a2 * kout[s]).sum(), np.sqrt(a2).mean(), len(s)))
    return np.array(out)
print('A. плоская волна (весь срез t ≈ −10, вектор (1, 0)): средняя |ψ| на срезах t =', times.tolist())
for eps in (0.0, 0.25, 0.5, 1.0):
    psi = propagate(w, LK, R, {k: np.array([1, 0], complex) for k in band0}, eps=eps, unit=True)
    s = slices(psi); print(f'  ε = {eps}: ' + ' '.join(f'{v:.3f}' for v in s[:, 3]))
print('B. пакет (гаусс σ = 2 по x на срезе t ≈ −10): отношение к первому срезу')
for eps in (0.0, 0.5, 1.0):
    init = {k: np.array([math.exp(-X[k, 1] ** 2 / 8), 0], complex) for k in band0 if abs(X[k, 1]) < 7}
    psi = propagate(w, LK, R, init, eps=eps, unit=True)
    s = slices(psi)
    for j, name in ((0, 'Σ|ψ|²'), (1, 'Σ k_вх|ψ|²'), (2, 'Σ k_исх|ψ|²')):
        print(f'  ε = {eps}, {name}: ' + ' '.join(f'{v / s[0, j]:.2f}' for v in s[:, j]))
    sys.stdout.flush()
