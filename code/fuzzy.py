#!/usr/bin/env python3
"""РАЗМЫТЫЙ КРАЙ ТЕНИ. Событие существует с вероятностью P(s), s — радиус тени:
P = 1 при s < r(1−w), линейно спадает до 0 при s = r(1+w). При w = 0 — чистый кристалл.
Доля «монеточных» событий задаёт случайную часть дрожи: ν = E[P(1−P)]/E[P].
Проверки: (1) средние в ромбах при бустах — без смещения; (2) дрожь в системе
кристалла и в общей системе при n до 10⁵; (3) размерность."""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
import gold4test as G
from gold4 import coord_values, ST, SI, r_for_density
rs = np.random.default_rng(8); t0 = time.time()

def points_shadow(box, rmax):
    vals = []
    for k, (lo, hi) in enumerate(box):
        if k == 0: c, cp = coord_values(lo / ST, hi / ST, rmax / SI); vals.append((c * ST, (cp * SI) ** 2))
        else: c, cp = coord_values(lo, hi, rmax); vals.append((c, cp ** 2))
    P = np.zeros((1, 4)); W = np.zeros(1)
    for k, (c, w2) in enumerate(vals):
        m = len(P); P = np.repeat(P, len(c), 0); W = np.repeat(W, len(c))
        P[:, k] = np.tile(c, m); W += np.tile(w2, m); keep = W <= rmax ** 2; P, W = P[keep], W[keep]
    return P, np.sqrt(W)

def prob(s, r, w):
    if w == 0: return (s <= r).astype(float)
    return np.clip((r * (1 + w) - s) / (2 * w * r), 0, 1)

def r_for(w):
    """радиус, при котором ожидаемая плотность = 1: ∫P(s) dV_4(s) = объём шара r0"""
    r0 = r_for_density(1.0)
    if w == 0: return r0
    # объём с весом: ∫ P(s) 4 s³ ds / r0⁴ = 1 — решаем численно
    def dens(r):
        s = np.linspace(0, r * (1 + w), 4001); return np.trapezoid(prob(s, r, w) * 4 * s ** 3, s) / r0 ** 4
    lo, hi = 0.5 * r0, 1.5 * r0
    for _ in range(60):
        mid = 0.5 * (lo + hi); (lo, hi) = (mid, hi) if dens(mid) < 1 else (lo, mid)
    return mid

def nu(r, w):
    s = np.linspace(0, r * (1 + w), 4001); P = prob(s, r, w); wt = 4 * s ** 3
    return np.trapezoid(P * (1 - P) * wt, s) / np.trapezoid(P * wt, s)

rnd = np.array([.48, -.71, .52]); rnd /= np.linalg.norm(rnd)
for w in (0.0, 0.1, 0.3):
    r = r_for(w); v = nu(r, w); rmax = r * (1 + w)
    print(f'w = {w}: r = {r:.3f}, доля «монеточной» дрожи ν = {v:.3f} (ожидаемая случайная часть σ/√n ≈ {math.sqrt(v):.2f})')
    for name, eta, n in (('покой', 0.0, np.array([1., 0, 0])), ('буст 1.5', 1.5, rnd)):
        row = []
        for nexp, M in ((3000, 40), (30000, 30), (100000, 20)):
            tau0 = (24 * nexp / math.pi) ** 0.25; G.tau0 = tau0; c = []
            for m in range(M):
                a = rs.uniform(0, 500, 4); b = a + tau0 * G.boost_vec(eta, n)
                Pp, s = points_shadow(G.interval_box(a, b, eta, n), rmax)
                keep = rs.random(len(Pp)) < prob(s, r, w)
                c.append(int(G.inside(Pp[keep], a, b).sum()))
            c = np.array(c); row.append(f'n={nexp}: среднее/n {c.mean()/nexp:.3f}, σ/√n {c.std(ddof=1)/math.sqrt(nexp):.2f}')
        print(f'   {name:9s} ' + ' | '.join(row) + f'   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
