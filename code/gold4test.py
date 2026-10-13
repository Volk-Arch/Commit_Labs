#!/usr/bin/env python3
"""Проверки золотого квазикристалла 3+1 (gold4.py).
1. Счёт точек в причинном ромбе объёма V (ожидание ρV = 30) у наблюдателей
   разной скорости: квазикристалл, целочисленная решётка Z⁴, Пуассон.
2. Различители порядка в ромбе против засева 4d.
"""
import sys, math, time
import numpy as np
sys.path.insert(0, '/home/claude')
from gold4 import points_in_box, r_for_density
from core import frac_pairs, mm_dim, height, links_of, interval_counts, spectral

rs = np.random.default_rng(8)
RHO = 1.0; NEXP = 30.0
tau0 = (24 * NEXP / (math.pi * RHO)) ** 0.25
r = r_for_density(RHO)

def boost_vec(eta, n):
    return np.concatenate([[math.cosh(eta)], math.sinh(eta) * n])

def interval_box(p, q, eta, n):
    # бокс, содержащий ромб: вершины p, q и сфера посередине (в системе ромба)
    mid = (p + q) / 2
    u = rs.normal(size=(200, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    # точка сферы в покое: (0, (τ0/2) u); бустируем вдоль n
    g = math.cosh(eta); bg = math.sinh(eta)
    par = u @ n
    pts = np.zeros((200, 4))
    pts[:, 0] = bg * (tau0 / 2) * par
    pts[:, 1:] = (tau0 / 2) * (u + ((g - 1) * par)[:, None] * n)
    pts = np.vstack([pts + mid, p, q])
    lo, hi = pts.min(0) - 0.3, pts.max(0) + 0.3
    return list(zip(lo, hi))

def inside(X, p, q):
    a = X - p; b = q - X
    return (a[:, 0] > np.linalg.norm(a[:, 1:], axis=1)) & (b[:, 0] > np.linalg.norm(b[:, 1:], axis=1))

def count_gold(p, q, box):
    return int(inside(points_in_box(box, r), p, q).sum())

def count_z4(p, q, box):
    g = [np.arange(math.ceil(lo), math.floor(hi) + 1) for lo, hi in box]
    X = np.stack(np.meshgrid(*g, indexing='ij'), -1).reshape(-1, 4).astype(float)
    return int(inside(X, p, q).sum())

if __name__ == '__main__':
    t0 = time.time()
    print(f'1. РОМБ ОБЪЁМА V (ожидание {NEXP:.0f} точек), τ0={tau0:.2f}.  Пуассон: 30 ± 5.5 при любой скорости.')
    dirs = {'вдоль оси x': np.array([1.0, 0, 0]),
            'случайное': (lambda v: v / np.linalg.norm(v))(np.array([0.48, -0.71, 0.52]))}
    for dname, n in dirs.items():
        print(f'  направление буста: {dname}')
        for eta in (0.0, 1.0, 2.0, 3.0):
            cg, cz = [], []
            M = 60 if eta < 3 else 30
            for m in range(M):
                p = rs.uniform(0, 500, 4); q = p + tau0 * boost_vec(eta, n)
                box = interval_box(p, q, eta, n)
                cg.append(count_gold(p, q, box)); cz.append(count_z4(p, q, box))
            cg, cz = np.array(cg), np.array(cz)
            print(f'    η={eta:.0f}: квазикристалл {cg.mean():5.1f} ± {cg.std():4.1f} (пустых {np.mean(cg==0):.2f})'
                  f'   |  Z⁴ {cz.mean():5.1f} ± {cz.std():5.1f} (пустых {np.mean(cz==0):.2f})   [{time.time()-t0:.0f} с]')
            sys.stdout.flush()
