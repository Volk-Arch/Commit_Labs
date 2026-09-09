#!/usr/bin/env python3
"""
ТРИ ТЕЛА: складываются ли влияния
=================================

У двух тел проверить нечего. У трёх — можно:
в слабом поле влияния СКЛАДЫВАЮТСЯ, и это проверяемо.

    вклад(A,B,C) - вклад(A)  =?=  [вклад(A,B)-вклад(A)] + [вклад(A,C)-вклад(A)]

Слева — настоящее тройное поле, справа — сумма парных.
"""
import argparse
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")      # Windows: консоль cp1251
    sys.stderr.reconfigure(encoding="utf-8")
import numpy as np
import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from twobody import local_density, metric, causal_order, direct_links, field_at


def build(bg, bodies, k, alpha, radius=0.2):
    X = np.vstack([bg] + bodies) if bodies else bg
    mask = np.zeros(len(X), dtype=bool)
    mask[:len(bg)] = True
    dens = local_density(X, radius)
    W = metric(dens, k, alpha)
    R = causal_order(X, W)
    return X, mask, R, direct_links(R)


def triangle(gap, k=4, seed=0, n_bg=2500, mass=200, spread=0.05):
    """Три тела в вершинах треугольника; точка замера — центр."""
    rs = np.random.default_rng(seed)
    bg = rs.random((n_bg, k))
    c = np.full(k, 0.5)
    # три направления, попарно как можно более разные
    dirs = []
    for i in range(3):
        v = np.zeros(k); v[i % k] = 1.0
        if k > 3: v[(i+1) % k] = -1.0
        dirs.append(v/np.linalg.norm(v))
    cs = [np.clip(c + gap*d, 0.05, 0.95) for d in dirs]
    bodies = [np.clip(x + rs.normal(0, spread, (mass, k)), 0, 1) for x in cs]
    return bg, bodies, c


def field_with(bg, bodies, subset, centre, k, alpha, nsample=120):
    X, mask, R, L = build(bg, [bodies[i] for i in subset], k, alpha)
    v = field_at(X, R, L, centre, mask, nsample)
    del X, R, L
    return v


def run(gap, seeds, k=4, alpha=1.5, n_bg=2500, mass=200):
    rows = []
    for s in range(seeds):
        bg, bodies, c = triangle(gap, k, 700 + s*137, n_bg, mass)
        f_a = field_with(bg, bodies, [0], c, k, alpha)
        f_ab = field_with(bg, bodies, [0, 1], c, k, alpha)
        f_ac = field_with(bg, bodies, [0, 2], c, k, alpha)
        f_abc = field_with(bg, bodies, [0, 1, 2], c, k, alpha)
        if not all(np.isfinite(v) for v in (f_a, f_ab, f_ac, f_abc)):
            continue
        d_b = f_ab - f_a               # вклад тела B
        d_c = f_ac - f_a               # вклад тела C
        d_bc = f_abc - f_a             # вклад обоих
        rows.append((f_a, f_ab, f_ac, f_abc, d_b, d_c, d_bc, d_b + d_c))
    return np.array(rows) if rows else None
