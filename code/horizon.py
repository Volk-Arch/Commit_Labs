#!/usr/bin/env python3
"""ГОРИЗОНТ ПО ПОРЯДКУ. Радиальная плоскость Шварцшильда (t, r) в координатах
Эддингтона–Финкельштейна (v, r):  ds² = −f dv² + 2 dv dr,  f = 1 − 2M/r.
  * объём √−g = 1: густота событий ОДНА И ТА ЖЕ везде (g_tt·g_rr = −1 —
    у Эйнштейна масса не меняет «кто существует»);
  * местное «раньше»: Δv > 0 и Δr < f(r)·Δv/2 (конус между падающим и
    уходящим светом). Внутри r < 2M f < 0 — все будущие направления
    ведут внутрь: горизонт возникает САМ, его никто не вписывал;
  * дальнее «раньше» — цепочкой близких (поиск в ширину).
Прибор: часы на неподвижном радиусе r_e (испускатель) и r = 6M (приёмник).
Задержка прихода по ч асам v сравнивается с точной:
  Δv = 2·(r*(R) − r*(r_e)),   r* = r + 2M·ln(r/2M − 1)   — расходится при r_e → 2M.
"""
import sys, math, time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

import os
M = 1.0; R = float(os.environ.get('RR', 6.0)); V0, V1 = 0.0, float(os.environ.get('V1', 42.0)); R0, R1 = 1.4, R + 0.6
EPS_V = EPS_R = float(os.environ.get('EPS', 0.45))
def f(r): return 1 - 2 * M / r
def rstar(r): return r + 2 * M * math.log(r / (2 * M) - 1)

def build(rho, rs, emit_radii, mass=True):
    n = rs.poisson(rho * (V1 - V0) * (R1 - R0))
    P = np.column_stack([rs.uniform(V0, V1, n), rs.uniform(R0, R1, n)])
    lines = []; NT = 1200
    for r in list(emit_radii) + [R]:
        vv = np.linspace(V0, V1, NT); lines.append(np.column_stack([vv, np.full(NT, r)]))
    Q = np.vstack([P] + lines); lab = np.concatenate([np.full(n, -1)] + [np.full(NT, k) for k in range(len(lines))])
    o = np.argsort(Q[:, 0]); Q = Q[o]; lab = lab[o]
    v, r = Q[:, 0], Q[:, 1]; rows, cols = [], []
    for i in range(len(Q)):
        j1 = np.searchsorted(v, v[i] + 1e-12); j2 = np.searchsorted(v, v[i] + EPS_V)
        if j2 <= j1: continue
        dv = v[j1:j2] - v[i]; dr = r[j1:j2] - r[i]
        fm = f(0.5 * (r[i] + r[j1:j2])) if mass else 1.0
        ok = (dr < fm * dv / 2) & (dr > -EPS_R)
        js = np.arange(j1, j2)[ok]; rows.extend([i] * len(js)); cols.extend(js)
    A = csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(len(Q), len(Q)))
    return Q, lab, A

def escape(Q, lab, A, k_emit, k_recv):
    em = np.flatnonzero(lab == k_emit); rc = np.flatnonzero(lab == k_recv)
    em = em[(Q[em, 0] > 1.0) & (Q[em, 0] < 6.0)][::6]
    out = []
    for e in em:
        reach = breadth_first_order(A, e, directed=True, return_predecessors=False)
        hit = rc[np.isin(rc, reach)]
        out.append(Q[hit, 0].min() - Q[e, 0] if len(hit) else np.inf)
    return np.array(out)

if __name__ == '__main__':
    rho = float(sys.argv[1]) if len(sys.argv) > 1 else 600
    radii = [float(x) for x in os.environ.get('RADII', '1.8,2.05,2.1,2.2,2.4,2.7,3.2,4.0').split(',')]
    t0 = time.time()
    rs = np.random.default_rng(1)
    Q, lab, A = build(rho, rs, radii, mass=True)
    rs = np.random.default_rng(1)
    Q0, lab0, A0 = build(rho, rs, radii, mass=False)     # те же точки, конусы плоские
    print(f'событий {(lab < 0).sum()}, рёбер {A.nnz} (плоский {A0.nnz})   [{time.time()-t0:.0f} с]')
    kR = len(radii)
    print(f"{'r_e/M':>6}{'приход Δv, порядок':>20}{'плоский порядок':>17}{'точно (Шварцшильд)':>20}{'плоский точно':>15}{'избыток: порядок / точно':>26}")
    for k, re_ in enumerate(radii):
        a = escape(Q, lab, A, k, kR); a0 = escape(Q0, lab0, A0, k, kR)
        exact = 2 * (rstar(R) - rstar(re_)) if re_ > 2 * M else math.inf
        flat = 2 * (R - re_)
        fin = np.isfinite(a)
        ma = a[fin].mean() if fin.any() else math.inf
        print(f'{re_:>6.2f}{ma:>14.2f} ({fin.mean()*100:3.0f}% дошло){a0.mean():>11.2f}{exact:>20.2f}{flat:>15.2f}'
              f'{(ma - a0.mean()):>14.2f} / {exact - flat:<8.2f}  [{time.time()-t0:.0f} с]')
        sys.stdout.flush()
