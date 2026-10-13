#!/usr/bin/env python3
"""Лаборатория «рисуем с нуля»: рост из одного события + бинарное правило.

Каждый шаг: предлагается прошлое нового события (каждый существующий
элемент берётся прямым родителем с вероятностью κ/n, прошлое = замыкание
вниз). Правило отвечает ДА/НЕТ. НЕТ -> новое предложение (до TRY раз).

Ничего, кроме порядка: ни координат, ни метрики. Сравнение — с засевом
4d-ромба того же N (тот же набор признаков, разброс по засевам).
"""
import math, sys, time
import numpy as np
sys.path.insert(0, '/home/claude')
from core import frac_pairs, mm_dim, height, links_of, interval_counts, spectral

TRY = 3000

def grow(N, kappa, rule, rs):
    past = [0]; hgt = [1]; lk = [0]
    while len(past) < N:
        n = len(past); ok = False
        for _ in range(TRY):
            q = min(0.5, kappa / n)
            S = np.flatnonzero(rs.random(n) < q)
            Sbits = 0; U = 0
            for s in S:
                Sbits |= 1 << int(s); U |= past[s]
            P = Sbits | U
            links = Sbits & ~U
            L = bin(links).count('1'); sz = bin(P).count('1')
            h = 1 + max((hgt[s] for s in S), default=0)
            ctx = dict(L=L, P=sz, h=h, n=n, links=links, prev_links=lk[-1], prev_past=past[-1])
            if rule(ctx):
                ok = True; break
        if not ok:
            return None, n
        past.append(P); hgt.append(h); lk.append(links)
    return past, N

def relmat(past):
    N = len(past); R = np.zeros((N, N), bool)
    for i, p in enumerate(past):
        b = bin(p)[2:][::-1]
        idx = [j for j, c in enumerate(b) if c == '1']
        R[i, idx] = True
    return R

def feats(R):
    f = frac_pairs(R); n = interval_counts(R, 3); N = len(R)
    return np.array([f, mm_dim(f) if 0 < f < 1 else np.nan, height(R)[0],
                     n[0] / N, n[1] / max(n[0], 1), n[2] / max(n[0], 1), spectral(R)])
NAMES = ['доля', 'разм', 'высота', 'связ/соб', 'n1/n0', 'n2/n0', 'спектр']

def sprinkle_R(N, rs):
    out = []
    while sum(len(o) for o in out) < N:
        c = rs.uniform(-1, 1, (4 * N, 4))
        ok = np.abs(c[:, 0]) + np.linalg.norm(c[:, 1:], axis=1) <= 1
        out.append(c[ok])
    X = np.concatenate(out)[:N]; X = X[np.argsort(X[:, 0])]
    dt = X[:, None, 0] - X[None, :, 0]; dx = np.linalg.norm(X[:, None, 1:] - X[None, :, 1:], axis=-1)
    R = dt > dx; np.fill_diagonal(R, False); return R

