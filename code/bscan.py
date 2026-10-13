"""Есть ли beta, где и доля 0.10, и спектр положительный."""
import math
import numpy as np
from core import (mm_dim, height, spectral, transitivity_violations,
                  frac_pairs, links_of)
from growth2 import grow

N, S = 1000, 6
print(f'desc, N={N}, eps=0.2, K=24, p=0.05, засевов {S}')
print('ЭТАЛОН конуса при N=1200: доля 0.1005, размерн. 3.995, '
      'высота 10.2, спектр +0.0185')
print('ящик: спектр -0.0050; перколяция: -0.10')
print()
print(f"{'beta':>7}{'доля пар':>17}{'размерность':>15}{'высота':>8}"
      f"{'спектр':>19}{'наруш':>7}{'связ/соб':>10}")
for b in (0.0, 1.0, 3.0, 10.0, 30.0, 300.0):
    acc = []
    for t in range(S):
        rs = np.random.default_rng(6600+17*t)
        R, ps = grow(N, 'desc', 0.05, b, 0.2, 24, rs)
        f = frac_pairs(R)
        acc.append((f, mm_dim(f), height(R)[0], spectral(R),
                    transitivity_violations(R), links_of(R).sum()/N))
        del R
    A = np.array(acc, float); m = A.mean(0); e = A.std(0)/math.sqrt(S)
    sig = m[3]/max(e[3], 1e-12)
    print(f'{b:>7.0f}{m[0]:>10.4f} ±{e[0]:<6.4f}{m[1]:>9.3f} ±{e[1]:<5.3f}'
          f'{m[2]:>8.1f}{m[3]:>+11.4f} ±{e[3]:<6.4f}{int(m[4]):>7}'
          f'{m[5]:>10.1f}   {sig:+.1f}σ')
