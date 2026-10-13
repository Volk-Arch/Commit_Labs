"""Калибровка: насколько «свет из цепочек» медленнее настоящего в плоском 2+1 (ρ = 30, ε = 1.2)."""
import sys, math, numpy as np
sys.path.insert(0, '.')
from chains import local_edges
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order
rng = np.random.default_rng(5); RHO, EPS, T, W = 30, 1.2, 45.0, 22.0
n = rng.poisson(RHO * T * (2 * W) * 6)
P = np.column_stack([rng.uniform(0, T, n), rng.uniform(-2, 2 * W - 2, n), rng.uniform(-3, 3, n)])
P = np.vstack([[0, 0, 0], P]); o = np.argsort(P[:, 0], kind='stable'); inv = np.empty_like(o); inv[o] = np.arange(len(o)); P = P[o]
ok = lambda mid, d: d[:, 0] ** 2 > d[:, 1] ** 2 + d[:, 2] ** 2
ei, ej = local_edges(P, ok, EPS, slab=2.0)
G = csr_matrix((np.ones(len(ei), bool), (ei, ej)), shape=(len(P), len(P)))
reach = breadth_first_order(G, int(inv[0]), directed=True, return_predecessors=False)
for D in (5, 10, 20, 30, 38):
    cell = reach[(np.abs(P[reach, 1] - D) < 0.2) & (np.abs(P[reach, 2]) < 0.2)]
    t = P[cell, 0].min(); print(f'расстояние {D}: первое прибытие {t:.2f}, отношение {t/D:.3f}')
