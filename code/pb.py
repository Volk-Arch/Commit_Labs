"""Есть ли точка (p, beta), где доля 0.10 И спектр положительный."""
import math
import numpy as np
from core import mm_dim, height, spectral, frac_pairs, links_of
from growth2 import grow

N, S, EPS = 1000, 3, 0.2
print(f'desc, N={N}, eps={EPS}, K=24, засевов {S}')
print('ЦЕЛЬ (конус при N=1000): доля 0.10, разм. 4.00, спектр +0.018,')
print('высота ~10, связей/соб. ~17')
print()
print(f"{'p':>6}{'beta':>6}{'доля пар':>17}{'разм.':>8}{'высота':>7}"
      f"{'спектр':>19}{'связ/соб':>9}{'оценка':>9}")
best = []
for p in (0.02, 0.05, 0.10, 0.20, 0.40):
    for b in (0.0, 3.0, 10.0, 30.0):
        acc = []
        for t in range(S):
            rs = np.random.default_rng(8800+17*t)
            R, _ = grow(N, 'desc', p, b, EPS, 24, rs)
            f = frac_pairs(R)
            acc.append((f, mm_dim(f), height(R)[0], spectral(R),
                        links_of(R).sum()/N))
            del R
        A = np.array(acc,float); m = A.mean(0); e = A.std(0)/math.sqrt(S)
        # расстояние до цели в своих единицах
        d = (abs(m[0]-0.10)/0.01 + abs(m[1]-4.0)/0.05
             + abs(m[3]-0.018)/0.005 + abs(m[2]-10)/2)
        best.append((d, p, b, m, e))
        print(f'{p:>6.2f}{b:>6.0f}{m[0]:>10.4f} ±{e[0]:<6.4f}{m[1]:>8.2f}'
              f'{m[2]:>7.1f}{m[3]:>+11.4f} ±{e[3]:<6.4f}{m[4]:>9.1f}'
              f'{d:>9.1f}')
best.sort()
print()
print('БЛИЖЕ ВСЕГО К ЦЕЛИ:')
for d,p,b,m,e in best[:3]:
    print(f'  p={p} beta={b:g}: доля {m[0]:.4f}, разм. {m[1]:.2f}, '
          f'высота {m[2]:.1f}, спектр {m[3]:+.4f} ±{e[3]:.4f}, '
          f'связей {m[4]:.1f}  (оценка {d:.1f})')
