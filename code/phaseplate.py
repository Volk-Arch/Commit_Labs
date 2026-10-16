#!/usr/bin/env python3
"""ФАЗОВАЯ ПЛАСТИНКА и две нормировки.

Нормировки (обе местные):
  split — событие делит полученное поровну между прямыми потомками (вес 1/исходящих);
  past  — «событие есть то, что из его прошлого»: амплитуда события = среднее по его прямым
          предкам (вес 1/входящих). Тогда |Z| ≤ 1 и модуль — согласие всех историй, ведущих сюда.
Светоделитель — лучшая пара портов кристалла (u ≈ 1), он фиксирован. Настройка — фазовая
пластинка: лишняя фаза α на втором пути (событие A₂) перед делителем. Угол непрерывный.
CHSH — максимум по α Алисы (2 шт.) и Боба (2 шт.), сигнал меряется.
"""
import sys, math, itertools, numpy as np
from scipy.optimize import minimize
from universe import golden_flat, poisson_flat, build_order
from phasesum import links_of
from beamsplit import smax, nearest

def amp_sum(w, L, src, phase, norm):
    n = w.n; Z = np.zeros(n, complex); Z[src] = 1
    I, J = np.nonzero(L); o = np.argsort(J, kind='stable'); I, J = I[o], J[o]
    deg = np.maximum(L.sum(1), 1) if norm == 'split' else np.maximum(L.sum(0), 1)
    wt = np.exp(1j * phase(I, J)) / (deg[I] if norm == 'split' else deg[J])
    st = np.searchsorted(J, np.arange(n)); en = np.searchsorted(J, np.arange(n), side='right')
    for j in range(src + 1, n):
        a, b = st[j], en[j]
        if a < b: Z[j] = (Z[I[a:b]] * wt[a:b]).sum()
    return Z

def experiment(make, seed, norm, k=math.pi, sig_max=0.02):
    w = make(4.0, 16.0, np.random.default_rng(seed)); R = build_order(w, 'proj'); L = links_of(R); X = w.X
    ph = lambda I, J: k * ((w.W[J] - w.W[I]) ** 2).sum(1)
    c0 = np.array([-7.0, 0.0]); tau = 7.0
    src = [j for j in range(w.n) if abs(X[j, 0] - c0[0]) <= 0.45 and abs(X[j, 1]) <= 3.0]
    A = [nearest(X, c0 + tau * np.array([math.cosh(e), math.sinh(e)])) for e in (0.25, 0.75)]
    B = [nearest(X, c0 + tau * np.array([math.cosh(e), -math.sinh(e)])) for e in (0.25, 0.75)]
    za, zb = [], []
    for r in src:
        Z = amp_sum(w, L, r, ph, norm); za.append([Z[a] for a in A]); zb.append([Z[b] for b in B])
    za, zb = np.array(za), np.array(zb); M = np.einsum('ri,rj->ij', za, zb)
    psi = M.reshape(4) / np.linalg.norm(M); S_ideal = smax(np.outer(psi, psi.conj()))
    C = 2 * abs(np.linalg.det(M)) / (np.abs(M) ** 2).sum()
    rho_mix = sum(np.outer(np.kron(a, b), np.kron(a, b).conj()) for a, b in zip(za, zb)); rho_mix /= np.trace(rho_mix).real
    def best_split(side, paths):
        Zp = [amp_sum(w, L, p, ph, norm) for p in paths]
        cand = [j for j in range(w.n) if 5.0 <= X[j, 0] <= 7.0 and side * X[j, 1] > 1.0 and all(R[p, j] for p in paths)]
        allp = []
        for p, q in itertools.combinations(cand, 2):
            T = np.array([[Zp[0][p], Zp[1][p]], [Zp[0][q], Zp[1][q]]]); nrm = (np.abs(T) ** 2).sum()
            if nrm > 0: allp.append((2 * abs(np.linalg.det(T)) / nrm, T))
        allp.sort(key=lambda t: -t[0]); return allp[:NSPLIT]
    SA = best_split(+1, A); SB = best_split(-1, B)
    out = None
    for (uA, TA), (uB, TB) in itertools.product(SA, SB):
        r = plate(M, TA, TB, seed, sig_max)
        if out is None or r[0] > out['S']: out = dict(S=r[0], sig=r[1], ang=r[2], uA=uA, uB=uB)
    return dict(C=C, S_ideal=S_ideal, S_mix=smax(rho_mix), **out)

NSPLIT = 1
def plate(M, TA, TB, seed, sig_max, restarts=None):
    restarts = restarts or RESTARTS
    def P(al, be):
        amp = (TA @ np.diag([1, np.exp(1j * al)])) @ M @ (TB @ np.diag([1, np.exp(1j * be)])).T
        p = np.abs(amp) ** 2; return p / p.sum()
    E = lambda p: p[0, 0] - p[0, 1] - p[1, 0] + p[1, 1]
    def score(v):
        a0, a1, b0, b1 = v; Ps = [[P(a0, b0), P(a0, b1)], [P(a1, b0), P(a1, b1)]]
        sig = max(abs(Ps[0][0].sum(1) - Ps[0][1].sum(1)).max(), abs(Ps[1][0].sum(1) - Ps[1][1].sum(1)).max(),
                  abs(Ps[0][0].sum(0) - Ps[1][0].sum(0)).max(), abs(Ps[0][1].sum(0) - Ps[1][1].sum(0)).max())
        return abs(E(Ps[0][0]) + E(Ps[0][1]) + E(Ps[1][0]) - E(Ps[1][1])), sig
    best = (0, 1, None); rs = np.random.default_rng(seed)
    for _ in range(restarts):
        res = minimize(lambda v: -score(v)[0] + 40 * max(0, score(v)[1] - sig_max), rs.uniform(0, 2 * math.pi, 4), method='Nelder-Mead', options=dict(maxiter=2000))
        S, sig = score(res.x)
        if sig <= sig_max and S > best[0]: best = (S, sig, np.degrees(res.x) % 360)
    return best
RESTARTS = 60

if __name__ == '__main__':
    if '--choose' in sys.argv: NSPLIT, RESTARTS = 8, 8
    SIG = float(sys.argv[sys.argv.index('--sig') + 1]) if '--sig' in sys.argv else 0.02
    SEEDS = range(31, 43) if '--many' in sys.argv else range(31, 37)
    for norm in (('past',) if '--choose' in sys.argv else ('past', 'split')):
        for name, make in (('кристалл', golden_flat), ('засев', poisson_flat)):
            rows = []
            for seed in SEEDS:
                r = experiment(make, seed, norm, sig_max=SIG); rows.append(r)
            S = np.array([r['S'] for r in rows]); Si = np.array([r['S_ideal'] for r in rows]); C = np.array([r['C'] for r in rows])
            print(f'нормировка «{norm}», {name}: C {np.round(C, 2).tolist()}; S идеальный {np.round(Si, 3).tolist()}; '
                  f'S с пластинкой {np.round(S, 3).tolist()} (сигнал ≤ {max(r["sig"] for r in rows):.3f}); делители u {[(round(r["uA"], 2), round(r["uB"], 2)) for r in rows]}; '
                  f'место записано ≤ {max(r["S_mix"] for r in rows):.3f}'); sys.stdout.flush()
