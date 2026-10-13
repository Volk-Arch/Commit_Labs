"""Самоусредняется ли выросший мир: падает ли разброс спектра с N."""
import math, numpy as np
from core import mm_dim, height, spectral, frac_pairs
from growth2 import grow

S = 12
print(f'разброс спектра по {S} засевам. Эталоны: конус с.к.о. 0.0038 / '
      f'0.0026 / 0.0016')
print('при N = 400 / 800 / 1600, то есть падает примерно как 1/sqrt(N)')
print()
print(f"{'правило':>14}{'N':>7}{'доля':>9}{'разм.':>8}"
      f"{'<спектр>':>11}{'с.к.о.':>9}{'мин':>9}{'макс':>9}{'размах':>9}")
for gen, p, b in (('desc', 0.10, 30.0), ('desc', 0.05, 0.0)):
    for N in (400, 800, 1600):
        v, f, d = [], [], []
        for t in range(S):
            rs = np.random.default_rng(31000 + 977*t)
            R, _ = grow(N, gen, p, b, 0.2, 24, rs)
            ff = frac_pairs(R)
            v.append(spectral(R)); f.append(ff); d.append(mm_dim(ff)); del R
        v = np.array(v)
        print(f'{f"{gen} p={p} b={b:g}":>14}{N:>7}{np.mean(f):>9.4f}'
              f'{np.mean(d):>8.2f}{v.mean():>+11.4f}{v.std():>9.4f}'
              f'{v.min():>+9.4f}{v.max():>+9.4f}{v.max()-v.min():>9.4f}',
              flush=True)
    print()
