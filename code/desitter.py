#!/usr/bin/env python3
"""ГИПОТЕЗА «РОСТ ВГЛУБЬ»: если мир переходит в себя при x -> τx (самоподобие),
то тень обязана меняться как r(T) ∝ 1/|T|: τx ∈ W ⟺ 0.618|x'| ≤ r(τT).
Плотность ∝ r⁴ ∝ T⁻⁴ = a(T)⁴ с a ∝ 1/|T| — это де Ситтер (плоское сечение,
конформное время T<0): у него растяжение (T,x) -> λ(T,x) — симметрия.
ПРЕДСКАЗАНИЕ, проверяемое по порядку: собственное время на каждый шаг τ
одинаково, т.е. самая длинная цепь от T до T/τ не зависит от T.
  де Ситтер: L₂/L₁ = 1;   плоский (r = const): L₂/L₁ = (|T₁|−|T₂|)/(|T₀|−|T₁|)·… = 0.618;
  излучение (a ∝ T, для сравнения): своё число.
"""
import sys, math, time, numpy as np
sys.path.insert(0, '/home/claude'); sys.path.insert(0, '.')
from gold4 import coord_values, r_for_density, ST, SI, TAU
from core import height
T0 = -3.0; T1 = T0 / TAU; T2 = T0 / TAU ** 2
HALF = (T2 - T0) / 2; MID = (T0 + T2) / 2

def enumerate_box(box, rmax):
    vals = []
    for k, (lo, hi) in enumerate(box):
        if k == 0: c, cp = coord_values(lo / ST, hi / ST, rmax / SI); vals.append((c * ST, (cp * SI) ** 2))
        else: c, cp = coord_values(lo, hi, rmax); vals.append((c, cp ** 2))
    P = np.zeros((1, 4)); W = np.zeros(1)
    for k, (c, w2) in enumerate(vals):
        m = len(P); P = np.repeat(P, len(c), 0); W = np.repeat(W, len(c))
        P[:, k] = np.tile(c, m); W += np.tile(w2, m); keep = W <= rmax ** 2; P, W = P[keep], W[keep]
    return P, W

def world(kind, rho0, rs, quasi=True):
    rfun = {'deSitter': lambda T: (abs(T0) / np.abs(T)),
            'плоский': lambda T: np.ones_like(T),
            'излучение': lambda T: (np.abs(T) / abs(T0)) ** -1 * 0 + (np.abs(T2) / np.abs(T)) ** -1 * (abs(T2)/abs(T0)) ** 0}[kind]
    if kind == 'излучение':      # a ∝ |T|^-? — для сравнения: a растёт к будущему как (T−T0+c) линейно
        rfun = lambda T: 1 + (T - T0) / (T2 - T0) * (abs(T0) / abs(T2) - 1)
    c = rs.uniform(0, 50, 3)
    box = [(T0, T2)] + [(c[k] - HALF, c[k] + HALF) for k in range(3)]
    r0 = r_for_density(rho0)
    if quasi:
        rmax = r0 * rfun(np.array([T2]))[0]
        P, W = enumerate_box(box, rmax)
        P[:, 1:] -= c
        P = P[W <= (r0 * rfun(P[:, 0])) ** 2]
    else:
        dmax = rho0 * rfun(np.array([T2]))[0] ** 4; vol = (T2 - T0) * (2 * HALF) ** 3
        n = rs.poisson(dmax * vol)
        X = np.column_stack([rs.uniform(T0, T2, n), rs.uniform(-HALF, HALF, (n, 3))])
        P = X[rs.random(n) < rho0 * rfun(X[:, 0]) ** 4 / dmax]
    P = P[np.abs(P[:, 0] - MID) + np.linalg.norm(P[:, 1:], axis=1) <= HALF]
    return P

def chain(P, a, b):
    """самая длинная цепь между событиями, ближайшими к (a,0,0,0) и (b,0,0,0)"""
    ia = np.argmin(np.linalg.norm(P - [a, 0, 0, 0], axis=1)); ib = np.argmin(np.linalg.norm(P - [b, 0, 0, 0], axis=1))
    pa, pb = P[ia], P[ib]
    def fut(p, Q): d = Q - p; return d[:, 0] > np.linalg.norm(d[:, 1:], axis=1)
    sel = fut(pa, P) & fut(P[ib][None].repeat(1, 0)[0], P) if False else fut(pa, P) & (lambda d: d[:, 0] > np.linalg.norm(d[:, 1:], axis=1))(pb - P)
    Q = np.vstack([pa, P[sel], pb]); Q = Q[np.argsort(Q[:, 0])]
    dt = Q[:, None, 0] - Q[None, :, 0]; dx = np.linalg.norm(Q[:, None, 1:] - Q[None, :, 1:], axis=-1)
    R = dt > dx; np.fill_diagonal(R, False)
    return height(R)[0] - 1

if __name__ == '__main__':
    rho0 = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    print(f'конформное время: T₀={T0:.3f}, T₁=T₀/τ={T1:.3f}, T₂=T₀/τ²={T2:.3f};  плотность при T₀: {rho0}')
    t0 = time.time()
    for kind in ('deSitter', 'плоский'):
        for quasi in (True, False):
            r = []
            for s in range(12):
                P = world(kind, rho0 if kind != 'плоский' else rho0 * 8, np.random.default_rng(10 + s), quasi)
                L1, L2 = chain(P, T0 + 0.02, T1), chain(P, T1, T2 - 0.02)
                r.append((len(P), L1, L2))
            r = np.array(r, float); q = r[:, 2] / r[:, 1]
            print(f'  {kind:10s} {"квазикристалл" if quasi else "засев":14s} событий {r[:,0].mean():6.0f}  цепь T₀→T₁ {r[:,1].mean():5.1f}  '
                  f'T₁→T₂ {r[:,2].mean():5.1f}  отношение {q.mean():.3f} ± {q.std(ddof=1)/math.sqrt(len(q)):.3f}   [{time.time()-t0:.0f} с]')
            sys.stdout.flush()
    print('ожидание: де Ситтер 1.000 (равное собственное время на каждый шаг τ);  плоский 0.618 (меньше из-за края)')
