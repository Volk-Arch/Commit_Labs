"""Читается ли доля античастиц из геометрии порядка?

Отдельные знаки из порядка не восстанавливаются (точность 0.51).
Но размерность менялась с долей. Если управляющий параметр —
вероятность попасть на пару разных знаков, 2f(1-f), то кривая
должна быть симметрична относительно f = 0.5.
"""
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from antimatter import rel_charged, closure
from core import dimension, height

N, K, SEEDS = 1200, 4, 5
fs = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print(f'N={N}, k={K}, семян {SEEDS}')
print(f"{'доля анти':>11}{'2f(1-f)':>10}{'размерность':>22}{'высота':>14}{'доля пар':>11}")
res = []
for f in fs:
    ds, hs, frs = [], [], []
    for s in range(SEEDS):
        rs = np.random.default_rng(3000 + 101 * s)
        X = rs.random((N, K))
        q = np.where(rs.random(N) < f, -1, 1)
        C = closure(rel_charged(X, q, 'B'))
        ds.append(dimension(R=C))
        hs.append(height(C)[0])
        frs.append(C.sum() / (N * (N - 1)))
    d, dsd = float(np.mean(ds)), float(np.std(ds))
    h, hsd = float(np.mean(hs)), float(np.std(hs))
    res.append((f, d, dsd))
    print(f'{f:>11.2f}{2*f*(1-f):>10.3f}'
          f'{f"{d:.3f} ± {dsd:.3f}":>22}'
          f'{f"{h:.1f} ± {hsd:.1f}":>14}'
          f'{float(np.mean(frs)):>11.4f}')

print()
print('ПРОВЕРКА СИММЕТРИИ относительно 0.5')
print(f"{'пара f':>14}{'разность размерн.':>20}{'в сигмах':>10}")
for a, b in ((0.05, 0.95), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.0, 1.0)):
    ra = next(r for r in res if abs(r[0] - a) < 1e-9)
    rb = next(r for r in res if abs(r[0] - b) < 1e-9)
    diff = ra[1] - rb[1]
    sig = np.hypot(ra[2], rb[2]) / np.sqrt(SEEDS)
    print(f'{f"{a} / {b}":>14}{diff:>20.4f}{abs(diff)/sig if sig > 0 else 0:>10.2f}')
