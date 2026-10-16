#!/usr/bin/env python3
"""SJ: срезка мод, плотность пакета и сверка с вероятностями делителя.
1) срезка по правилу Афшорди–Асланбейги–Соркина: оставить λ > √N/(4π);
2) пакет: вероятность на срезах, центр и ширина;
3) делитель на оси x = 0 (точное зеркало P): вероятности портов по SJ (плотность Лёвдина на срезе портов)
   против |T|² из правила (та же линейная версия правила) — согласуются ли «два правила Борна»."""
import sys, math, numpy as np
from scipy.linalg import eigh
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
from sj import sj, slice_prob
sys.argv = _a
rho, Lw = 4.0, 12.0
w = CH.sym_world(rho, Lw); R = CH.build_order(w, 'proj'); LK = CH.prep(w, CH.links_of(R), 1); X = w.X; n = w.n
idx = lambda evs: np.array([2 * k + m for k in evs for m in (0, 1)])
for eps in (0.0, 0.5, 1.0):
    Rp, iD, lam, V, Wfull = sj(w, LK, eps)
    order = np.argsort(-lam); cut = math.sqrt(n) / (4 * math.pi)
    kA = int((lam > cut).sum())
    print(f'\nε = {eps}: N = {n}, порог ААС √N/4π = {cut:.2f} → мод {kA}')
    for kk in (kA, 2 * kA, 4 * kA):
        sel = order[:kk]; Wk = (V[:, sel] * lam[sel]) @ V[:, sel].conj().T
        src = [k for k in range(n) if -8 <= X[k, 0] <= -7 and abs(X[k, 1]) < 4]
        f = np.zeros(2 * n, complex)
        for k in src: f[2 * k] = math.exp(-X[k, 1] ** 2 / 4)
        out = []
        for t0 in (-6.0, -3.0, 0.0, 3.0, 6.0):
            S_ev = [k for k in range(n) if t0 - 0.5 <= X[k, 0] <= t0 + 0.5]
            P, dens, rk = slice_prob(Wk, f, idx(S_ev), rcond=1e-9)
            dx = dens.reshape(-1, 2).sum(1); xs = X[S_ev, 1]; cx = (dx * xs).sum() / dx.sum(); sx = math.sqrt((dx * (xs - cx) ** 2).sum() / dx.sum())
            out.append(f'{P:.3f} ({cx:+.1f}±{sx:.1f})')
        print(f'  мод {kk}: пакет, P_S (центр ± ширина) на t = −6, −3, 0, 3, 6: ' + ' | '.join(out))
        # делитель на оси: A1 около (−3, 1.2), A2 = P(A1); порты на срезе t ≈ 3
        A1 = int(np.argmin(((X - np.array([-3.0, 1.2])) ** 2).sum(1))); A2 = CH.img(w, A1, CH.P, CH.P)
        port_ev = [k for k in range(n) if abs(X[k, 0] - 3) <= 0.5 and R[A1, k] and R[A2, k]]
        S = idx(port_ev)
        rows = []
        for m0 in (0, 1):
            for A, mm in ((A1, m0), (A2, 1 - m0)):
                e = np.zeros(2 * n, complex); e[2 * A + mm] = 1
                P, dens, rk = slice_prob(Wk, e, S, rcond=1e-9)
                ruleZ = Rp[S, 2 * A + mm]
                rows.append((P, dens, np.abs(ruleZ) ** 2))
        # согласие формы распределения по срезу портов: корреляция SJ-плотности и |Z правила|² (нормированных)
        cors = [np.corrcoef(d / d.sum(), z / z.sum())[0, 1] for P, d, z in rows]
        print(f'      делитель: P(найти на срезе портов) = {", ".join(f"{r[0]:.3f}" for r in rows)}; '
              f'совпадение формы SJ-плотности с |ψ правила|² по срезу: {", ".join(f"{c:.2f}" for c in cors)}')
    sys.stdout.flush()
