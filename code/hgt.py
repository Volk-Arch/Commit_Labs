"""Вес по ВЫСОТЕ: структура без обратной связи.

Число потомков растёт, когда к событию присоединяются — самоусиление.
Высота события при добавлении максимального элемента НЕ меняется, значит
отбор по высоте структурен, но обратной связи не создаёт. Если
воспроизводимость останется на уровне перколяции (с.к.о. ~0.002), а
спектр сдвинется в плюс, цель достигнута.
"""
import math, numpy as np
from core import mm_dim, height as H, spectral, frac_pairs, links_of
from weight import w_of

def grow_h(N, gamma, p, rs, K=8):
    R = np.zeros((N,N), bool); hh = np.zeros(N, np.int64)
    for k in range(1, N):
        w = (hh[:k].astype(np.float64)+1.0)**gamma
        w /= w.sum()
        m = min(max(1,int(p*k)), k)
        s = np.zeros(k, bool); s[rs.choice(k, m, replace=False, p=w)] = True
        A = R[:k,:k].astype(np.float32)
        d = s | ((A.T @ s.astype(np.float32)) > 0)
        R[k,:k] = d
        hh[k] = (hh[:k][d].max()+1) if d.any() else 0
    return R

S = 8
print(f'вес затравки ~ (1+высота)^gamma, beta=0, N=600, засевов {S}')
print('ЦЕЛЬ: спектр около +0.010 при с.к.о. ~0.003, доля ~0.10, разм. 4.00')
print('gamma=0 — перколяция (эталон воспроизводимости)')
print()
print(f"{'gamma':>7}{'p':>6}{'доля':>9}{'разм.':>7}{'высота':>7}"
      f"{'<спектр>':>11}{'с.к.о.':>9}{'связ/соб':>9}")
for g in (-2.0, -1.0, 0.0, 1.0, 2.0, 4.0):
    for p in (0.03, 0.06):
        v,f,d,h,l = [],[],[],[],[]
        for t in range(S):
            rs = np.random.default_rng(52000+977*t)
            R = grow_h(600, g, p, rs); ff = frac_pairs(R)
            v.append(spectral(R)); f.append(ff); d.append(mm_dim(ff))
            h.append(H(R)[0]); l.append(links_of(R).sum()/600); del R
        v=np.array(v)
        print(f'{g:>7.1f}{p:>6.2f}{np.mean(f):>9.4f}{np.mean(d):>7.2f}'
              f'{np.mean(h):>7.1f}{v.mean():>+11.4f}{v.std():>9.4f}'
              f'{np.mean(l):>9.1f}', flush=True)
