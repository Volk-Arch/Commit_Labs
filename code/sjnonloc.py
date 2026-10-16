#!/usr/bin/env python3
"""НЕЛОКАЛЬНОСТЬ SJ-ВАКУУМА. Алиса поворачивает пластинку (фаза на звеньях у себя). Меняется ли что-то у Боба
(пространственно разнесённая область слева)? Коммутатор iΔ в области Боба меняться не может (пути между
событиями Боба не проходят через Алису). А вакуум W = положительная часть iΔ строится по всему миру сразу."""
import sys, math, numpy as np
_a = sys.argv; sys.argv = ['x', '0.5', '1']
src = open('/home/claude/code/bellsj2.py').read(); src = src[:src.index('angles = np.linspace')]
exec(compile(src, 'bellsj2_setup', 'exec'))
sys.argv = _a
Bev_all = [k for k in range(n) if 3.5 <= X[k, 0] <= 6.0 and X[k, 1] < -2.0 and not any(R[a, k] or R[k, a] for a in [A1, A2] + Aev)]
Bi = np.array([2 * k + m for k in Bev_all for m in (0, 1)])
out = {}
for al in (0.0, math.pi / 2, math.pi):
    fld = np.where(cutA, np.exp(1j * al), 1.0)
    Rp, iD, lam, V, W = sj(w, LK, eps, fld)
    out[al] = (iD[np.ix_(Bi, Bi)], W[np.ix_(Bi, Bi)], Rp[np.ix_(Bi, Bi)])
for al in (math.pi / 2, math.pi):
    dD = np.abs(out[al][0] - out[0.0][0]).max() / np.abs(out[0.0][0]).max()
    dW = np.abs(out[al][1] - out[0.0][1]).max() / np.abs(out[0.0][1]).max()
    dR = np.abs(out[al][2] - out[0.0][2]).max() / np.abs(out[0.0][2]).max()
    dn = np.abs(np.diag(out[al][1]).real - np.diag(out[0.0][1]).real).max() / np.diag(out[0.0][1]).real.max()
    print(f'пластинка Алисы {math.degrees(al):.0f}°: в области Боба ({len(Bev_all)} событий) меняется: пропагатор R {dR:.1e}, '
          f'коммутатор iΔ {dD:.1e}, вакуум W {dW:.1e} (плотность «частиц» W(b,b) {dn:.1e})')
