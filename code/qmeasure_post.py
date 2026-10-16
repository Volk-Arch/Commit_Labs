#!/usr/bin/env python3
"""Разбор qmeasure: почему с отбором S мал. Сравниваем детекторы:
 (a) область 15 событий, вероятности складываются по событиям (каждое событие — своя запись);
 (b) одно событие-порт (как раньше);
 (c) «мода»: амплитуды по событиям области складываются когерентно (детектор не знает, где именно)."""
import itertools, math, numpy as np
out = np.load('/home/claude/data/qmeasure_out.npy', allow_pickle=True).item()
NG = int(round(math.sqrt(len(out)))); nD = 15
def rows(block, i=None, c=None):
    base = {'Ap': 0, 'Aq': nD, 'Bp': 2 * nD, 'Bq': 3 * nD}[block]
    if i is None: return [2 * (base + k) + cc for k in range(nD) for cc in (0, 1)]
    return [2 * (base + i) + c]
def best(Pt, tol):
    E = Pt[..., 0, 0] - Pt[..., 0, 1] - Pt[..., 1, 0] + Pt[..., 1, 1]; mA = Pt[..., 0, :].sum(-1); mB = Pt[..., :, 0].sum(-1); bS = 0
    for a0, a1 in itertools.combinations(range(NG), 2):
        for b0, b1 in itertools.combinations(range(NG), 2):
            sig = max(abs(mA[a0, b0] - mA[a0, b1]), abs(mA[a1, b0] - mA[a1, b1]), abs(mB[a0, b0] - mB[a1, b0]), abs(mB[a0, b1] - mB[a1, b1]))
            if sig > tol: continue
            e = (E[a0, b0], E[a0, b1], E[a1, b0], E[a1, b1])
            bS = max(bS, abs(e[0] + e[1] + e[2] - e[3]), abs(e[0] - e[1] + e[2] + e[3]), abs(e[0] + e[1] - e[2] + e[3]), abs(-e[0] + e[1] + e[2] + e[3]))
    return bS
def table(fn):
    Pt = np.zeros((NG, NG, 2, 2))
    for (ia, ib), r in out.items():
        P = fn(r['Psi_det']); Pt[ia, ib] = P / P.sum()
    return Pt
inc = lambda Psi: np.array([[(np.abs(Psi[np.ix_(rows(a), rows(b))]) ** 2).sum() for b in ('Bp', 'Bq')] for a in ('Ap', 'Aq')])
print(f'(a) области, сумма вероятностей по событиям: S = {best(table(inc), 0.005):.3f}')
bestS = (0, None)
for i in range(nD):
    for c in (0, 1):
        A = [rows('Ap', i, c)[0], rows('Aq', i, 1 - c)[0]]; B = [rows('Bp', i, 1 - c)[0], rows('Bq', i, c)[0]]
        f = lambda Psi: np.abs(Psi[np.ix_(A, B)]) ** 2
        Pt = table(f)
        if not np.isfinite(Pt).all(): continue
        S = best(Pt, 0.005)
        if S > bestS[0]: bestS = (S, (i, c), best(Pt, 0.001))
print(f'(b) одно событие-порт, лучшее из {2 * nD}: S = {bestS[0]:.3f} (≤ 0.5 %), {bestS[2]:.3f} (≤ 0.1 %)')
def mode(Psi):
    out_ = np.zeros((2, 2))
    for ai, a in enumerate(('Ap', 'Aq')):
        for bi, b in enumerate(('Bp', 'Bq')):
            tot = 0
            for ca in (0, 1):
                for cb in (0, 1):
                    ra = [2 * ({'Ap': 0, 'Aq': nD}[a] + k) + ca for k in range(nD)]; rb = [2 * ({'Bp': 2 * nD, 'Bq': 3 * nD}[b] + k) + cb for k in range(nD)]
                    tot += abs(Psi[np.ix_(ra, rb)].sum()) ** 2
            out_[ai, bi] = tot
    return out_
print(f'(c) «мода» (когерентная сумма по событиям области): S = {best(table(mode), 0.005):.3f}')
