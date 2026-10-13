"""Две обязательные проверки точки p=0.1, beta=30: по N и по размеру пула."""
import math
import numpy as np
from core import (mm_dim, height, spectral, frac_pairs, links_of,
                  transitivity_violations)
from growth2 import grow

P, B, EPS, S = 0.10, 30.0, 0.2, 3
def run(N, K, seeds=S):
    acc=[]
    for t in range(seeds):
        rs = np.random.default_rng(9900+17*t)
        R,_ = grow(N, 'desc', P, B, EPS, K, rs); f = frac_pairs(R)
        acc.append((f, mm_dim(f), height(R)[0], spectral(R),
                    transitivity_violations(R), links_of(R).sum()/N)); del R
    A=np.array(acc,float); return A.mean(0), A.std(0)/math.sqrt(seeds)

print(f'desc, p={P}, beta={B:g}, eps={EPS}, засевов {S}')
print('ЭТАЛОН конуса: доля 0.100 устойч., разм. 4.00 устойч., спектр>0,')
print('высота ~N^0.25 (8.5@600, 10.2@1200), связей ~N^0.75')
print()
print('ПРОВЕРКА 1: УСТОЙЧИВОСТЬ ПО РАЗМЕРУ МИРА (K=24)')
print(f"{'N':>7}{'доля пар':>17}{'разм.':>15}{'высота':>8}{'спектр':>19}"
      f"{'наруш':>7}{'связ/соб':>9}")
ns, hs, ls = [], [], []
for N in (400, 800, 1600, 2400):
    m,e = run(N, 24)
    ns.append(N); hs.append(m[2]); ls.append(m[5])
    print(f'{N:>7}{m[0]:>10.4f} ±{e[0]:<6.4f}{m[1]:>8.2f} ±{e[1]:<5.2f}'
          f'{m[2]:>8.1f}{m[3]:>+11.4f} ±{e[3]:<6.4f}{int(m[4]):>7}{m[5]:>9.1f}',
          flush=True)
lg = np.log(ns)
print(f'  показатель высоты по N: {np.polyfit(lg,np.log(hs),1)[0]:.3f} '
      f'(цель 0.25)   связей: {np.polyfit(lg,np.log(ls),1)[0]:.3f} (цель 0.75)')
print()
print('ПРОВЕРКА 2: ЗАВИСИТ ЛИ ОТ РАЗМЕРА ПУЛА (N=1000)')
print(f"{'K':>7}{'доля пар':>17}{'разм.':>15}{'высота':>8}{'спектр':>19}")
for K in (8, 16, 24, 48, 96):
    m,e = run(1000, K)
    print(f'{K:>7}{m[0]:>10.4f} ±{e[0]:<6.4f}{m[1]:>8.2f} ±{e[1]:<5.2f}'
          f'{m[2]:>8.1f}{m[3]:>+11.4f} ±{e[3]:<6.4f}', flush=True)
