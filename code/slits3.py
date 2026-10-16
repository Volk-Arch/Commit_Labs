#!/usr/bin/env python3
"""ДВЕ ЩЕЛИ С ПРАВИЛОМ «ПОВОРОТ = i» (без фазы тени).
Источник в (−12, 0); щели — пара событий S₁ и S₂ = P(S₁) при t ≈ −4 на расстоянии d; экран — события
при t ≈ 6. Амплитуда на экране — сумма вкладов через S₁ и через S₂ (каждый — распространение из щели
с тем 2-вектором, что до неё дошёл). Интенсивность |ψ₀|² + |ψ₁|².
Сравниваем с некогерентной суммой и с предсказанием «частицы массы m»: разность фаз m·(τ₁ − τ₂),
τ_i — собственное время от щели i до точки экрана; m — частота из dirac.py.
"""
import sys, math, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
from prop import propagate
sys.argv = _a

w = CH.sym_world(4.0, 24.0); R = CH.build_order(w, 'proj'); LK = CH.prep(w, CH.links_of(R), 1); X = w.X
src = int(np.argmin(((X - np.array([-12.0, 0.0])) ** 2).sum(1)))
MASS = {0.0: 0.0, 0.5: 0.291, 1.0: 0.491}       # из dirac.py, ρ = 4
for eps in (0.0, 0.5, 1.0):
    P0 = propagate(w, LK, R, {src: np.array([1, 0], complex)}, eps=eps, unit=True)
    for d in (2.0, 4.0, 6.0):
        S1 = int(np.argmin(((X - np.array([-4.0, d / 2])) ** 2).sum(1))); S2 = CH.img(w, S1, CH.P, CH.P)
        scr = np.nonzero((np.abs(X[:, 0] - 6) < 0.45) & (np.abs(X[:, 1]) < 9) & R[S1] & R[S2])[0]
        scr = scr[np.argsort(X[scr, 1])]
        Q1 = propagate(w, LK, R, {S1: P0[S1]}, eps=eps, unit=True)[scr]
        Q2 = propagate(w, LK, R, {S2: P0[S2]}, eps=eps, unit=True)[scr]
        Icoh = (np.abs(Q1 + Q2) ** 2).sum(1); Iinc = (np.abs(Q1) ** 2).sum(1) + (np.abs(Q2) ** 2).sum(1)
        r = Icoh / Iinc - 1                     # интерференционный член (от −1 до 1)
        vis = (np.percentile(r, 95) - np.percentile(r, 5)) / 2
        y = X[scr]; t1 = np.sqrt((y[:, 0] - X[S1, 0]) ** 2 - (y[:, 1] - X[S1, 1]) ** 2); t2 = np.sqrt((y[:, 0] - X[S2, 0]) ** 2 - (y[:, 1] - X[S2, 1]) ** 2)
        pred = np.cos(MASS[eps] * (t1 - t2))
        corr = np.corrcoef(r, pred)[0, 1] if eps > 0 else float('nan')
        # число полос на экране (смены знака r вдоль x, сглаженное окном 5)
        rs = np.convolve(r, np.ones(5) / 5, mode='same'); nfr = int((np.diff(np.sign(rs)) != 0).sum() / 2)
        print(f'ε = {eps}, d = {d} (щели x = ±{X[S1,1]:.2f}): точек экрана {len(scr)}; видность {vis:.3f}; полос ~{nfr}; '
              f'совпадение с cos(m·Δτ): r = {corr:.3f}'); sys.stdout.flush()
