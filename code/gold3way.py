#!/usr/bin/env python3
"""Три проверки золотого квазикристалла 3+1 за один проход.

A. ОТПЕЧАТОК: мешает ли «комковатость» малых интервалов действию БД и
   крупным интервалам (профиль накопленный, действие на событие).
B. КРИВИЗНА КАК ТЕНЬ: радиус шара тени зависит от времени, r ∝ a(η),
   значит плотность ∝ a⁴ — это Вселенная Фридмана с излучением, a = η
   (причинность конформно-плоской метрики ТОЧНО плоская, кривизна вся в
   плотности). Проверка: самая длинная цепь от нижней вершины до события
   на оси обязана идти как собственное время Фридмана (η² − η₁²)/2, а не
   как плоское (η − η₁). Контроль — засев с той же плотностью и плоский.
C. РОСТ: каждое событие несёт 8 целых чисел. «Раньше» считается ТОЧНО,
   целочисленной арифметикой в Z[τ], без единой плавающей точки; события
   рождаются в случайном допустимом порядке и каждое само проверяет
   свою тень. Мир не должен зависеть от порядка рождения.
"""
import sys, math, time
import numpy as np
sys.path.insert(0, '/home/claude'); sys.path.insert(0, '.')
from gold4 import coord_values, r_for_density, TAU, TP, ST, SI
from drawlab import sprinkle_R
from core import frac_pairs, mm_dim, height, links_of, interval_counts, bd_action

t0 = time.time()
def stamp(): return f'[{time.time()-t0:.0f} с]'

def order(P):
    dt = P[:, None, 0] - P[None, :, 0]; dx = np.linalg.norm(P[:, None, 1:] - P[None, :, 1:], axis=-1)
    R = dt > dx; np.fill_diagonal(R, False); return R

def labeled_box(box, rmax):
    """как points_in_box, но с целыми метками (a_i, b_i) и квадратом тени"""
    (Tl, Th), *sp = box
    axes = []
    for k, (lo, hi) in enumerate(box):
        if k == 0:
            # c = a + bτ, физическое T = ST c
            bmin = math.floor((lo / ST - rmax / SI) / 5 ** .5) - 1; bmax = math.ceil((hi / ST + rmax / SI) / 5 ** .5) + 1
            sc, si, rr = ST, SI, rmax / SI
        else:
            bmin = math.floor((lo - rmax) / 5 ** .5) - 1; bmax = math.ceil((hi + rmax) / 5 ** .5) + 1
            sc, si, rr = 1.0, 1.0, rmax
        A, B = [], []
        for b in range(bmin, bmax + 1):
            alo = max(math.ceil(lo / sc - b * TAU), math.ceil(-rr - b * TP))
            ahi = min(math.floor(hi / sc - b * TAU), math.floor(rr - b * TP))
            if ahi >= alo:
                a = np.arange(alo, ahi + 1); A.append(a); B.append(np.full(len(a), b))
        A, B = np.concatenate(A), np.concatenate(B)
        axes.append((A, B, sc * (A + B * TAU), (si * (A + B * TP)) ** 2))
    lab = np.zeros((1, 8), np.int64); P = np.zeros((1, 4)); W = np.zeros(1)
    for k, (A, B, c, w2) in enumerate(axes):
        n = len(c); m = len(P)
        lab = np.repeat(lab, n, 0); P = np.repeat(P, n, 0); W = np.repeat(W, n)
        lab[:, 2 * k] = np.tile(A, m); lab[:, 2 * k + 1] = np.tile(B, m)
        P[:, k] = np.tile(c, m); W += np.tile(w2, m)
        keep = W <= rmax * rmax; lab, P, W = lab[keep], P[keep], W[keep]
    return lab, P, W

# ---------------------------------------------------------------- A
ONLY_B = '--b' in sys.argv
if not ONLY_B: print('A. ОТПЕЧАТОК: действие БД на событие и накопленный профиль, плоский ромб')
V4 = 2 * math.pi / 3
for Nt in (() if ONLY_B else (1000, 2000)):
    r = r_for_density(Nt / V4); rs = np.random.default_rng(Nt)
    SG, PG = [], []
    for s in range(3):
        c = rs.uniform(0, 100, 4)
        lab, P, W = labeled_box([(c[k] - 1, c[k] + 1) for k in range(4)], r)
        P = P - c; m = np.abs(P[:, 0]) + np.linalg.norm(P[:, 1:], axis=1) <= 1
        R = order(P[m]); SG.append(bd_action(R) / len(R))
        A_ = R.astype(np.float32); h = np.bincount((A_ @ A_).astype(int)[R], minlength=200)[:200]; PG.append(np.cumsum(h) / R.sum())
    N = int(m.sum())
    SS, PS = [], []
    for s in range(6):
        R = sprinkle_R(N, np.random.default_rng(900 + s)); SS.append(bd_action(R) / len(R))
        A_ = R.astype(np.float32); h = np.bincount((A_ @ A_).astype(int)[R], minlength=200)[:200]; PS.append(np.cumsum(h) / R.sum())
    PG, PS = np.array(PG).mean(0), np.array(PS)
    d = np.abs(PG - PS.mean(0)); sd = PS.std(0, ddof=1) + 1e-9
    print(f'  N≈{N}: S/N квазикристалл {np.mean(SG):+.3f} ± {np.std(SG):.3f}   засев {np.mean(SS):+.3f} ± {np.std(SS, ddof=1):.3f}')
    for lo, hi in ((0, 3), (4, 10), (11, 40), (41, 199)):
        print(f'      накопленная доля, m {lo:>3}…{hi:<3}: макс. отличие {d[lo:hi+1].max():.4f}  ({(d[lo:hi+1]/sd[lo:hi+1]).max():.1f}σ)')
    print('     ', stamp()); sys.stdout.flush()

# ---------------------------------------------------------------- B
print('\nB. КРИВИЗНА КАК ТЕНЬ: Фридман с излучением, a(η) = η, плотность ρ₀·a⁴')
E1, E2 = 1.5, 3.0; HALF = (E2 - E1) / 2; MID = (E1 + E2) / 2
RHO0 = float([a for a in sys.argv[1:] if not a.startswith('--')][0]) if [a for a in sys.argv[1:] if not a.startswith('--')] else 150.0
def in_diamond(P): return np.abs(P[:, 0] - MID) + np.linalg.norm(P[:, 1:], axis=1) <= HALF
def frw_quasi(rmax_fn, rs):
    c = rs.uniform(0, 100, 4); c[0] = 0
    box = [(E1, E2)] + [(c[k] - HALF, c[k] + HALF) for k in (1, 2, 3)]
    rmax = rmax_fn(E2)
    lab, P, W = labeled_box(box, rmax)
    P = P.copy(); P[:, 1:] -= c[1:]
    keep = (W <= rmax_fn(P[:, 0]) ** 2) & in_diamond(P)
    return P[keep]
def frw_sprinkle(N, rs, weight):
    out = []
    while sum(len(o) for o in out) < N:
        X = np.column_stack([rs.uniform(E1, E2, 20 * N), rs.uniform(-HALF, HALF, (20 * N, 3))])
        X = X[in_diamond(X)]
        X = X[rs.random(len(X)) < weight(X[:, 0])]
        out.append(X)
    return np.concatenate(out)[:N]
def chain_profile(P):
    P = P[np.argsort(P[:, 0])]
    R = order(P); _, D = height(R)
    axis = np.linalg.norm(P[:, 1:], axis=1) < 0.15
    return P[axis, 0], D[axis].astype(float)
def judge(eta, h, lab):
    best = {}
    for name, f in (('Фридман (η²−η₁²)/2', (eta ** 2 - E1 ** 2) / 2), ('плоское η−η₁', eta - E1)):
        k = (f @ h) / (f @ f); res = h - k * f
        best[name] = res.std()
    print(f'  {lab:34s} событий на оси {len(h):4d}   разброс остатка: ' + '   '.join(f'{k}: {v:.3f}' for k, v in best.items()))
rs = np.random.default_rng(5)
rq = lambda eta: r_for_density(RHO0 * np.asarray(eta) ** 4 / 1.0) if np.ndim(eta) else r_for_density(RHO0 * eta ** 4)
rflat = lambda eta: r_for_density(RHO0 * MID ** 4) + 0 * np.asarray(eta)
for lab, gen in (('квазикристалл, r ∝ a (кривой)', lambda: frw_quasi(rq, rs)),
                 ('квазикристалл, r = const (плоский)', lambda: frw_quasi(rflat, rs))):
    acc_e, acc_h = [], []
    for s in range(3):
        P = gen(); e, h = chain_profile(P); acc_e.append(e); acc_h.append(h)
    print(f'     (событий в ромбе {len(P)})', end='')
    judge(np.concatenate(acc_e), np.concatenate(acc_h), lab)
Nq = len(frw_quasi(rq, rs))
acc_e, acc_h = [], []
for s in range(3):
    P = frw_sprinkle(Nq, np.random.default_rng(40 + s), lambda e: (e / E2) ** 4)
    e, h = chain_profile(P); acc_e.append(e); acc_h.append(h)
print(f'     (событий в ромбе {Nq})', end='')
judge(np.concatenate(acc_e), np.concatenate(acc_h), 'засев с плотностью ∝ a⁴ (кривой)')
print('     ', stamp()); sys.stdout.flush()

# ---------------------------------------------------------------- C
if ONLY_B: sys.exit()
print('\nC. РОСТ: точная арифметика в Z[τ] и случайный порядок рождения')
def zt_mul(p1, q1, p2, q2):           # (p1+q1τ)(p2+q2τ), τ² = τ + 1
    return p1 * p2 + q1 * q2, p1 * q2 + q1 * p2 + q1 * q2
def zt_sign(p, q):                    # знак p + qτ = (2p+q + q√5)/2, точно
    A = 2 * p + q; B = q
    s = np.where((A >= 0) & (B >= 0), 1, np.where((A <= 0) & (B <= 0), -1, 0))
    mix = s == 0
    big = np.sign(A * A - 5 * B * B) * np.sign(A)   # |A| против |B|√5
    s = np.where(mix, big, s)
    return np.where((A == 0) & (B == 0), 0, s)
def exact_earlier(lab_new, lab_old):
    """для нового события: какие старые раньше него (целочисленно)"""
    d = lab_new[None, :] - lab_old                        # разности меток
    pt, qt = d[:, 0], d[:, 1]
    p2, q2 = zt_mul(pt, qt, pt, qt)
    p2, q2 = zt_mul(p2, q2, 0, 1)                          # τ·Δt²
    for k in (1, 2, 3):
        a, b = zt_mul(d[:, 2 * k], d[:, 2 * k + 1], d[:, 2 * k], d[:, 2 * k + 1]); p2 -= a; q2 -= b
    return (zt_sign(pt, qt) > 0) & (zt_sign(p2, q2) > 0)
r = r_for_density(1200 / V4); c = np.full(4, 41.7)
lab, P, W = labeled_box([(c[k] - 1, c[k] + 1) for k in range(4)], r)
P = P - c; m = np.abs(P[:, 0]) + np.linalg.norm(P[:, 1:], axis=1) <= 1
lab, P = lab[m], P[m]; N = len(P)
Rf = order(P)
diffs = []
for trial in range(3):
    rs = np.random.default_rng(trial)
    # случайный допустимый порядок рождения: случайная линейная продолжение порядка
    key = rs.random(N); born = []
    indeg = Rf.sum(1).copy(); ready = list(np.flatnonzero(indeg == 0))
    while ready:
        j = ready.pop(int(rs.integers(len(ready)))); born.append(j)
        for k in np.flatnonzero(Rf[:, j]):
            indeg[k] -= 1
            if indeg[k] == 0: ready.append(k)
    born = np.array(born)
    Rg = np.zeros((N, N), bool)
    for t_, j in enumerate(born):                          # рождение по одному
        if t_: Rg[j, born[:t_]] = exact_earlier(lab[j], lab[born[:t_]])
    diffs.append(int((Rg ^ Rf).sum()))
print(f'  событий {N}; расхождений точного роста с плавающим счётом при трёх разных порядках рождения: {diffs}')
print('     ', stamp())
