#!/usr/bin/env python3
"""Два шага за проход.

1. СВОЁ ДЕЙСТВИЕ. Резкое БД смотрит на интервалы 0…3 — ровно там комки
   кристалла. Размазанное (прореживание с плотностью ε, smear-doc) берёт
   ВСЕ размеры с гладкими весами W_m(ε) — комки должны усредниться.
   Мера: S_ε/(εN) на квазикристалле (4 положения ромба) против засева (6),
   плоский ромб; затем ОТКЛИК НА КРИВИЗНУ: Фридман с веществом a = η²
   (R = 6a''/a³ ≠ 0) минус плоский — у квазикристалла и у засева.
2. ПРИЛИВНАЯ КРИВИЗНА ФОРМОЙ ТЕНИ. Тень — эллипсоид вместо шара (плотность
   та же). Ожидание по теореме (причинный порядок задаёт конформную
   геометрию, а конусы здесь плоские): приливов не будет, а сломается
   симметрия — появится выделенная система отсчёта. Мера — счёт в
   вытянутых ромбах, как в gold4test.
"""
import sys, math, time
import numpy as np
sys.path.insert(0, '/home/claude'); sys.path.insert(0, '.')
from gold4 import coord_values, r_for_density, ST, SI, TAU, TP
from drawlab import sprinkle_R
t0 = time.time()
W_BD = (-1.0, 9.0, -16.0, 8.0)
def smeared_w(eps, M):
    m = np.arange(M + 1); W = np.zeros(M + 1)
    for j in range(4):
        cm = np.array([math.comb(int(k), j) if k >= j else 0 for k in m], float)
        W += W_BD[j] * cm * eps ** j * (1 - eps) ** np.maximum(m - j, 0)
    return eps ** 2 * W
def hist(R):
    A = R.astype(np.float32); return np.bincount((A @ A).astype(np.int64)[R])
def S_eps(h, N, eps):
    W = smeared_w(eps, len(h) - 1); return (eps * N + W @ h) / (eps * N)
def order(P):
    dt = P[:, None, 0] - P[None, :, 0]; dx = np.linalg.norm(P[:, None, 1:] - P[None, :, 1:], axis=-1)
    R = dt > dx; np.fill_diagonal(R, False); return R
EPS = (1.0, 0.5, 0.2, 0.1, 0.05)

def box_points(box, r, scale=(1, 1, 1, 1)):
    vals = []
    for k, (lo, hi) in enumerate(box):
        if k == 0:
            c, cp = coord_values(lo / ST, hi / ST, r / SI / scale[0]); vals.append((c * ST, (cp * SI * scale[0]) ** 2))
        else:
            c, cp = coord_values(lo, hi, r / scale[k]); vals.append((c, (cp * scale[k]) ** 2))
    P = np.zeros((1, 4)); Wt = np.zeros(1)
    for k, (c, w2) in enumerate(vals):
        m = len(P); P = np.repeat(P, len(c), 0); Wt = np.repeat(Wt, len(c))
        P[:, k] = np.tile(c, m); Wt += np.tile(w2, m)
        keep = Wt <= r * r; P, Wt = P[keep], Wt[keep]
    return P

# ------------------------------------------------------------ 1a плоский
print('1. РАЗМАЗАННОЕ ДЕЙСТВИЕ S_ε/(εN), плоский ромб, N≈2000')
V4 = 2 * math.pi / 3; r = r_for_density(2000 / V4); rs = np.random.default_rng(3)
Q = []
for s in range(4):
    c = rs.uniform(0, 100, 4); P = box_points([(c[k] - 1, c[k] + 1) for k in range(4)], r) - c
    P = P[np.abs(P[:, 0]) + np.linalg.norm(P[:, 1:], axis=1) <= 1]; R = order(P)
    Q.append([S_eps(hist(R), len(R), e) for e in EPS])
N = len(R)
S = [[S_eps(hist(Rs), N, e) for e in EPS] for Rs in (sprinkle_R(N, np.random.default_rng(500 + s)) for s in range(6))]
Q, S = np.array(Q), np.array(S)
print(f"   {'ε':>5} | {'квазикристалл':>20} | {'засев':>20} | отличие в σ засева")
for k, e in enumerate(EPS):
    print(f'   {e:5.2f} | {Q[:,k].mean():+9.3f} ± {Q[:,k].std():7.3f} | {S[:,k].mean():+9.3f} ± {S[:,k].std(ddof=1):7.3f} | {(Q[:,k].mean()-S[:,k].mean())/S[:,k].std(ddof=1):+6.1f}')
print(f'   [{time.time()-t0:.0f} с]'); sys.stdout.flush()

# ------------------------------------------------------------ 1b кривизна
print('\n1б. ОТКЛИК НА КРИВИЗНУ: Фридман с веществом a = η² (η от 2 до 3) минус плоский')
E1, E2 = 2.0, 3.0; HALF = 0.5; MID = 2.5
def in_d(P): return np.abs(P[:, 0] - MID) + np.linalg.norm(P[:, 1:], axis=1) <= HALF
VOLC = math.pi / 24 * (2 * HALF) ** 4
def quasi_frw(aexp, Ntarget, rs):
    # плотность ρ₀ a⁴ = ρ₀ η^{4·aexp}, нормировка на ожидаемое N
    grid = np.linspace(E1, E2, 401); wt = np.maximum(HALF - np.abs(grid - MID), 0) ** 3
    mean_a4 = (grid ** (4 * aexp) * wt).sum() / wt.sum()
    rho0 = Ntarget / (VOLC * mean_a4)
    rmax = r_for_density(rho0 * E2 ** (4 * aexp))
    c = rs.uniform(0, 100, 4); c[0] = 0
    lab_box = [(E1, E2)] + [(c[k] - HALF, c[k] + HALF) for k in (1, 2, 3)]
    # перечисление с теневым квадратом
    vals = []
    for k, (lo, hi) in enumerate(lab_box):
        if k == 0:
            cc, cp = coord_values(lo / ST, hi / ST, rmax / SI); vals.append((cc * ST, (cp * SI) ** 2))
        else:
            cc, cp = coord_values(lo, hi, rmax); vals.append((cc, cp ** 2))
    P = np.zeros((1, 4)); Wt = np.zeros(1)
    for k, (cc, w2) in enumerate(vals):
        m = len(P); P = np.repeat(P, len(cc), 0); Wt = np.repeat(Wt, len(cc))
        P[:, k] = np.tile(cc, m); Wt += np.tile(w2, m)
        keep = Wt <= rmax ** 2; P, Wt = P[keep], Wt[keep]
    P[:, 1:] -= c[1:]
    keep = (Wt <= r_for_density(rho0 * P[:, 0] ** (4 * aexp)) ** 2) & in_d(P)
    return P[keep]
def sprinkle_frw(aexp, Ntarget, rs):
    out = []
    while sum(len(o) for o in out) < Ntarget:
        X = np.column_stack([rs.uniform(E1, E2, 30 * Ntarget), rs.uniform(-HALF, HALF, (30 * Ntarget, 3))])
        X = X[in_d(X)]; X = X[rs.random(len(X)) < (X[:, 0] / E2) ** (4 * aexp)]; out.append(X)
    return np.concatenate(out)[:Ntarget]
res = {}
for lab, gen in (('квазикристалл', quasi_frw), ('засев', sprinkle_frw)):
    for aexp in (0, 2):
        vals = []
        for s in range(4 if lab == 'квазикристалл' else 6):
            P = gen(aexp, 2000, np.random.default_rng(70 + s)); R = order(P[np.argsort(P[:, 0])])
            vals.append([S_eps(hist(R), len(R), e) for e in EPS])
        res[(lab, aexp)] = np.array(vals)
    print(f'   {lab}: N≈{len(R)}  [{time.time()-t0:.0f} с]'); sys.stdout.flush()
print(f"   {'ε':>5} | {'Δ квазикристалл':>20} | {'Δ засев':>20}")
for k, e in enumerate(EPS):
    out = []
    for lab in ('квазикристалл', 'засев'):
        a, b = res[(lab, 2)][:, k], res[(lab, 0)][:, k]
        out.append((a.mean() - b.mean(), math.hypot(a.std(ddof=1) / math.sqrt(len(a)), b.std(ddof=1) / math.sqrt(len(b)))))
    print(f'   {e:5.2f} | {out[0][0]:+9.3f} ± {out[0][1]:7.3f} | {out[1][0]:+9.3f} ± {out[1][1]:7.3f}')

# ------------------------------------------------------------ 2 форма тени
print('\n2. ФОРМА ТЕНИ: эллипсоид (время тени растянуто в 2 раза), та же плотность; ромб объёма 30')
import gold4test as G
rs2 = np.random.default_rng(12)
for lab, sc in (('шар', (1, 1, 1, 1)), ('эллипсоид 2:1:1:1', (0.5, 1, 1, 1)), ('эллипсоид 1:2:1:1', (1, 0.5, 1, 1))):
    rr = G.r * (np.prod(sc)) ** 0.25 if False else G.r / (np.prod(sc)) ** -0.25
    rr = G.r * np.prod(sc) ** 0.25
    line = []
    for dname, n in (('ось x', np.array([1., 0, 0])), ('ось y', np.array([0, 1., 0])), ('случ.', np.array([.48, -.71, .52]) / np.linalg.norm([.48, -.71, .52]))):
        for eta in (0.0, 3.0):
            cg = []
            for m in range(30):
                p = rs2.uniform(0, 500, 4); q = p + G.tau0 * G.boost_vec(eta, n)
                box = G.interval_box(p, q, eta, n)
                cg.append(int(G.inside(box_points(box, rr, sc), p, q).sum()))
            cg = np.array(cg); line.append(f'{dname} η={eta:.0f}: {cg.mean():4.1f}±{cg.std():4.1f}')
    print(f'   {lab:18s} ' + ' | '.join(line)); sys.stdout.flush()
print(f'   [{time.time()-t0:.0f} с]')
