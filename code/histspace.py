#!/usr/bin/env python3
"""ПУТЬ 2: ПРОСТРАНСТВО ИСТОРИЙ (Доукер–Джонстон–Соркин). Классы историй: «прошла через событие e (компонента m)»
для e в полосе t ≈ −3. Функционал декогеренции D(e, e′) = Σ_{x — конец истории} ψ_e(x)·ψ_{e′}(x)*, где концы —
максимальные события мира, обрезанного по t ≤ T_f. Грам-матрица D задаёт скалярное произведение пространства
историй. Вопрос: одно ли это пространство при разных T_f (то есть не зависит ли прошлое от длины будущего)?"""
import sys, math, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
sys.argv = _a
eps = 0.5
w = CH.sym_world(4.0, 16.0); R = CH.build_order(w, 'proj'); LK = CH.prep(w, CH.links_of(R), 1); X = w.X; n = w.n
I, J, lab, st, en = LK['I'], LK['J'], LK['lab'], LK['st'], LK['en']
s0, coin = 1 / math.sqrt(1 + eps * eps), 1j * eps
band = np.nonzero((np.abs(X[:, 0] + 3) < 0.3) & (np.abs(X[:, 1]) < 4))[0]
nc = 2 * len(band); psi = np.zeros((n, 2, nc), complex)
for c, k in enumerate(band): psi[k, 0, 2 * c] = 1; psi[k, 1, 2 * c + 1] = 1
t0 = X[band, 0].min(); inb = set(band.tolist())
for j in np.nonzero(X[:, 0] >= t0)[0]:
    if j in inb: continue
    a, b = st[j], en[j]
    if a >= b: continue
    ii, mm = I[a:b], lab[a:b]; n0, n1 = (mm != 1).sum(), (mm != 0).sum()
    c0 = s0 * (psi[ii, 0] + coin * psi[ii, 1]); c1 = s0 * (psi[ii, 1] + coin * psi[ii, 0])
    psi[j, 0] = c0[mm != 1].sum(0) / max(n0, 1); psi[j, 1] = c1[mm != 0].sum(0) / max(n1, 1)
def ends(Tf):
    ok = X[:, 0] <= Tf; has = np.zeros(n, bool); sel = ok[I] & ok[J]; has[I[sel]] = True
    return np.nonzero(ok & ~has & (X[:, 0] > X[band, 0].max()))[0]
G = {}
for Tf in (0.0, 3.0, 6.0, 9.0, 12.0):
    E = ends(Tf); A = psi[E].reshape(2 * len(E), nc)
    G[Tf] = A.conj().T @ A
    ev = np.linalg.eigvalsh(G[Tf]); r = (ev > 1e-10 * ev.max()).sum()
    print(f'T_f = {Tf:+.0f}: концов историй {len(E)}; след D = {np.trace(G[Tf]).real:.3f}; ранг пространства историй {r} из {nc}')
base = G[0.0] / np.trace(G[0.0]).real
for Tf in (3.0, 6.0, 9.0, 12.0):
    g = G[Tf] / np.trace(G[Tf]).real
    print(f'  сравнение T_f = 0 и {Tf:+.0f}: ‖D̂ − D̂₀‖/‖D̂₀‖ = {np.linalg.norm(g - base) / np.linalg.norm(base):.3f}; '
          f'вероятности «через e» (диагональ) меняются до {np.abs(np.diag(g) - np.diag(base)).max() / np.diag(base).real.mean():.0%} от средней')
