#!/usr/bin/env python3
"""SJ: движущийся пакет. Источник — гаусс на наклонном срезе (система, движущаяся со скоростью v = tanh 0.5).
Нетривиальная проверка плотности вероятности SJ: центр на лабораторных срезах должен идти по x = x₀ + v·t,
а полная вероятность на каждом срезе — оставаться 1."""
import sys, math, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
from sj import sj, slice_prob
sys.argv = _a
w = CH.sym_world(4.0, 12.0); R = CH.build_order(w, 'proj'); LK = CH.prep(w, CH.links_of(R), 1); X = w.X; n = w.n
idx = lambda evs: np.array([2 * k + m for k in evs for m in (0, 1)])
b = 0.5; v = math.tanh(b); ch, sh = math.cosh(b), math.sinh(b)
tp = ch * X[:, 0] - sh * X[:, 1]; xp = -sh * X[:, 0] + ch * X[:, 1]     # координаты движущейся системы
src = [k for k in range(n) if -7.5 <= tp[k] <= -6.5 and abs(xp[k]) < 4]
x0 = np.mean(X[src, 1]); t0s = np.mean(X[src, 0])
for eps in (0.0, 0.5):
    Rp, iD, lam, V, Wf = sj(w, LK, eps); order = np.argsort(-lam)
    f = np.zeros(2 * n, complex)
    for k in src: f[2 * k] = math.exp(-xp[k] ** 2 / 4)
    psi = Rp @ f
    for kk in (40, 80, None):
        Wk = Wf if kk is None else (V[:, order[:kk]] * lam[order[:kk]]) @ V[:, order[:kk]].conj().T
        T, C, Pl, Cr = [], [], [], []
        for t0 in (-4.0, -2.0, 0.0, 2.0, 4.0):
            S_ev = [k for k in range(n) if t0 - 0.5 <= X[k, 0] <= t0 + 0.5]
            P, dens, rk = slice_prob(Wk, f, idx(S_ev), rcond=1e-9)
            dx = dens.reshape(-1, 2).sum(1); xs = X[S_ev, 1]
            T.append(t0); C.append((dx * xs).sum() / dx.sum()); Pl.append(P)
            z = (np.abs(psi[idx(S_ev)]) ** 2).reshape(-1, 2).sum(1); Cr.append((z * xs).sum() / z.sum())
        sl = np.polyfit(T, C, 1)[0]; slr = np.polyfit(T, Cr, 1)[0]
        name = 'все моды' if kk is None else f'{kk} мод'
        print(f'ε = {eps}, {name}: P_S = {" ".join(f"{p:.3f}" for p in Pl)}; центр x = {" ".join(f"{c:+.2f}" for c in C)}; '
              f'скорость SJ {sl:.3f}, по |ψ|² правила {slr:.3f}, ожидание v = {v:.3f}')
    sys.stdout.flush()
