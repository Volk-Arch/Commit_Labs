#!/usr/bin/env python3
"""ИНТЕРФЕРОМЕТР ИЗ КРИСТАЛЛА: две фазовые пластинки и два светоделителя на каждой стороне.
Путь частицы: A₁, A₂ → пластинка α на A₂ → делитель T₁ (пара событий около t ≈ 4) → пластинка β на
втором выходе → делитель T₂ (пара событий около t ≈ 7) → детекторы. Итог T₂·diag(1,e^{iβ})·T₁·diag(1,e^{iα})
при делителях 50/50 даёт любое направление измерения. Нормировка «среднее прошлого», фаза π·σ².
Делители выбираются один раз на сторону (лучшие по качеству и балансу), настройки — только углы."""
import sys, math, itertools, numpy as np
from scipy.optimize import minimize
from universe import golden_flat, poisson_flat, build_order
from phasesum import links_of
from phaseplate import amp_sum
from beamsplit import smax, nearest

def pick_split(w, R, Zin, ins, tlo, thi, side, X):
    cand = [j for j in range(w.n) if tlo <= X[j, 0] <= thi and side * X[j, 1] > 0.5 and all(R[p, j] for p in ins)]
    best = (-1, None, None)
    for p, q in itertools.combinations(cand, 2):
        T = np.array([[Zin[0][p], Zin[1][p]], [Zin[0][q], Zin[1][q]]]); nrm = (np.abs(T) ** 2).sum()
        if nrm == 0: continue
        u = 2 * abs(np.linalg.det(T)) / nrm
        bal = 1 - abs(np.abs(T[0, 0]) ** 2 + np.abs(T[1, 1]) ** 2 - np.abs(T[0, 1]) ** 2 - np.abs(T[1, 0]) ** 2) / nrm
        if u * bal > best[0]: best = (u * bal, T, (p, q), u, bal)
    return best

def experiment(make, seed, sig_max=0.005, k=math.pi):
    w = make(4.0, 16.0, np.random.default_rng(seed)); R = build_order(w, 'proj'); L = links_of(R); X = w.X
    ph = lambda I, J: k * ((w.W[J] - w.W[I]) ** 2).sum(1)
    c0 = np.array([-7.0, 0.0]); tau = 7.0
    src = [j for j in range(w.n) if abs(X[j, 0] - c0[0]) <= 0.45 and abs(X[j, 1]) <= 3.0]
    A = [nearest(X, c0 + tau * np.array([math.cosh(e), math.sinh(e)])) for e in (0.25, 0.75)]
    B = [nearest(X, c0 + tau * np.array([math.cosh(e), -math.sinh(e)])) for e in (0.25, 0.75)]
    za, zb = [], []
    for r in src:
        Z = amp_sum(w, L, r, ph, 'past'); za.append([Z[a] for a in A]); zb.append([Z[b] for b in B])
    M = np.einsum('ri,rj->ij', np.array(za), np.array(zb))
    psi = M.reshape(4) / np.linalg.norm(M); S_ideal = smax(np.outer(psi, psi.conj()))
    def side_ops(paths, side):
        Zp = [amp_sum(w, L, p, ph, 'past') for p in paths]
        s1 = pick_split(w, R, Zp, paths, 3.5, 4.5, side, X)
        if s1[1] is None: return None
        p, q = s1[2]; Zq = [amp_sum(w, L, p, ph, 'past'), amp_sum(w, L, q, ph, 'past')]
        s2 = pick_split(w, R, Zq, [p, q], 6.5, 7.5, side, X)
        if s2[1] is None: return None
        return s1, s2
    oa, ob = side_ops(A, +1), side_ops(B, -1)
    if oa is None or ob is None: return None
    def unit(T):
        if not UNITARIZE: return T
        Wm, _, Vh = np.linalg.svd(T); return Wm @ Vh                # ближайший унитарный (те же фазы, без потерь)
    def U(ops, al, be):
        (_, T1, _, _, _), (_, T2, _, _, _) = ops
        T1, T2 = unit(T1), unit(T2)
        return T2 @ np.diag([1, np.exp(1j * be)]) @ T1 @ np.diag([1, np.exp(1j * al)])
    def P(v, i, j):
        amp = U(oa, v[2 * i], v[2 * i + 1]) @ M @ U(ob, v[4 + 2 * j], v[5 + 2 * j]).T
        p = np.abs(amp) ** 2; return p / p.sum()
    E = lambda p: p[0, 0] - p[0, 1] - p[1, 0] + p[1, 1]
    def score(v):
        Ps = [[P(v, i, j) for j in range(2)] for i in range(2)]
        sig = max(abs(Ps[0][0].sum(1) - Ps[0][1].sum(1)).max(), abs(Ps[1][0].sum(1) - Ps[1][1].sum(1)).max(),
                  abs(Ps[0][0].sum(0) - Ps[1][0].sum(0)).max(), abs(Ps[0][1].sum(0) - Ps[1][1].sum(0)).max())
        return abs(E(Ps[0][0]) + E(Ps[0][1]) + E(Ps[1][0]) - E(Ps[1][1])), sig
    best = (0, 1); rs = np.random.default_rng(seed)
    for _ in range(24):
        res = minimize(lambda v: -score(v)[0] + 60 * max(0, score(v)[1] - sig_max), rs.uniform(0, 2 * math.pi, 8), method='Nelder-Mead', options=dict(maxiter=4000))
        S, sig = score(res.x)
        if sig <= sig_max and S > best[0]: best = (S, sig)
    q = lambda o: (round(o[0][3], 2), round(o[0][4], 2), round(o[1][3], 2), round(o[1][4], 2))
    return dict(S_ideal=S_ideal, S=best[0], sig=best[1], qa=q(oa), qb=q(ob))

UNITARIZE = False
if __name__ == '__main__':
    UNITARIZE = '--unit' in sys.argv
    for name, make in (('кристалл', golden_flat), ('засев', poisson_flat)):
        rows = [r for r in (experiment(make, s) for s in range(31, 43)) if r]
        S = np.array([r['S'] for r in rows]); Si = np.array([r['S_ideal'] for r in rows])
        print(f'{name}: S идеальный {np.round(Si, 3).tolist()}\n   S интерферометр {np.round(S, 3).tolist()} (сигнал ≤ 0.005)\n'
              f'   доля от идеального {np.round((S - 2) / np.maximum(Si - 2, 1e-9), 2).tolist()}\n   делители (u, баланс) Алиса {[r["qa"] for r in rows][:4]}…'); sys.stdout.flush()
