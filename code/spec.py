"""Что делает спектр положительным: число шкал или их коррелированность."""
import math
import numpy as np
from core import rel_of, spectral, frac_pairs, mm_dim

N, S = 700, 4

def sphere(n, rs):
    u = rs.normal(size=(n, 3)); return u/np.linalg.norm(u,axis=1,keepdims=True)

def indep(k, rs):                 # k НЕЗАВИСИМЫХ шкал (ящик)
    return rel_of(rs.random((N, k)))

def cone(k, rs):                  # k КОРРЕЛИРОВАННЫХ шкал: s_u = t - u.x
    X = np.empty((N,4)); X[:,0]=rs.uniform(0,2,N); X[:,1:]=rs.uniform(-1,1,(N,3))
    U = sphere(k, rs)
    return rel_of(X[:,0:1] - X[:,1:] @ U.T)

print(f'N={N}, засевов {S}')
print('гипотеза: знак спектра отслеживает коррелированность шкал '
      '(конус), а не их число')
print()
print(f"{'k шкал':>8}{'':>4}{'НЕЗАВИСИМЫЕ (ящик)':>34}{'':>4}"
      f"{'КОРРЕЛИРОВАННЫЕ (конус)':>36}")
print(f"{'':>12}{'доля':>9}{'разм.':>8}{'спектр':>17}"
      f"{'':>4}{'доля':>9}{'разм.':>8}{'спектр':>17}")
for k in (2, 3, 4, 6, 10, 20, 40):
    row = []
    for fn in (indep, cone):
        acc = []
        for t in range(S):
            rs = np.random.default_rng(7700+17*t)
            R = fn(k, rs); f = frac_pairs(R)
            acc.append((f, mm_dim(f), spectral(R)))
            del R
        A = np.array(acc,float); row.append((A.mean(0), A.std(0)/math.sqrt(S)))
    (m1,e1),(m2,e2) = row
    print(f'{k:>8}{"":>4}{m1[0]:>9.4f}{m1[1]:>8.2f}'
          f'{m1[2]:>+10.4f} ±{e1[2]:<5.4f}{"":>4}'
          f'{m2[0]:>9.4f}{m2[1]:>8.2f}{m2[2]:>+10.4f} ±{e2[2]:<5.4f}')
