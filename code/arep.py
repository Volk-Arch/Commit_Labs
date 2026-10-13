"""Воспроизводимое предложение с конусной подписью: скан alpha и p при beta=0."""
import math, numpy as np
import growth2
from core import mm_dim, height, spectral, frac_pairs, links_of
from growth2 import grow

N, S = 600, 8
print(f'beta=0 (отбора нет, воспроизводимость наследуется), N={N}, '
      f'засевов {S}')
print(f'ЦЕЛЬ при N={N}: доля ~0.10, разм. 4.00, спектр около +0.010, '
      f'с.к.о. ~0.003')
print('ящик: -0.0069   конус(400): +0.0069   конус(800): +0.0145')
print()
print(f"{'alpha':>6}{'p':>6}{'доля':>9}{'разм.':>7}{'высота':>7}"
      f"{'<спектр>':>11}{'с.к.о.':>9}{'связ/соб':>9}{'оценка':>8}")
best=[]
for al in (0.0, 0.5, 1.0, 1.5, 2.0):
    growth2._ALPHA = al
    for p in (0.03, 0.05, 0.08):
        v,f,d,h,l = [],[],[],[],[]
        for t in range(S):
            rs = np.random.default_rng(41000+977*t)
            R,_ = grow(N, 'alpha', p, 0.0, 0.2, 8, rs)
            ff = frac_pairs(R)
            v.append(spectral(R)); f.append(ff); d.append(mm_dim(ff))
            h.append(height(R)[0]); l.append(links_of(R).sum()/N); del R
        v=np.array(v)
        sc = abs(np.mean(f)-0.10)/0.01 + abs(np.mean(d)-4.0)/0.05 \
             + abs(v.mean()-0.010)/0.003 + v.std()/0.003
        best.append((sc, al, p, np.mean(f), np.mean(d), v.mean(), v.std()))
        print(f'{al:>6.1f}{p:>6.2f}{np.mean(f):>9.4f}{np.mean(d):>7.2f}'
              f'{np.mean(h):>7.1f}{v.mean():>+11.4f}{v.std():>9.4f}'
              f'{np.mean(l):>9.1f}{sc:>8.1f}', flush=True)
best.sort()
print()
print('ЛУЧШИЕ ТРИ:')
for sc,al,p,f,d,m,s in best[:3]:
    print(f'  alpha={al} p={p}: доля {f:.4f}, разм. {d:.2f}, '
          f'спектр {m:+.4f} ±с.к.о.{s:.4f}  (оценка {sc:.1f})')
