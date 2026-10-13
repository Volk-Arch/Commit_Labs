"""D = прошлое ОДНОГО позднего элемента: единственная схема с линейным ростом.

|D| = |past(a)|+1. Если a брать среди поздних (их прошлое ~ вдвое больше
среднего), то P_{k+1} = P_k(1+1/(k+1)), то есть P ~ k линейно. Перекрытие
наследуется: D берётся из того же семейства прошлых, а не строится
объединением.

win — доля последних элементов, из которых берётся a.
extra — сколько дополнительных элементов добавить к затравке (0 = чистая схема).
"""
import math, numpy as np
from core import mm_dim, height, spectral, frac_pairs, links_of

def grow_one(N, win, extra, rs):
    R = np.zeros((N,N), bool)
    for k in range(1, N):
        lo = max(0, k - max(1, int(win*k)))
        sel = [int(rs.integers(lo, k))]
        for _ in range(extra):
            sel.append(int(rs.integers(0, k)))
        s = np.zeros(k, bool); s[sel] = True
        A = R[:k,:k].astype(np.float32)
        R[k,:k] = s | ((A.T @ s.astype(np.float32)) > 0)
    return R

def jprof(R):
    A = R.astype(np.float32); inter = A@A.T; sz = A.sum(1)
    un = sz[:,None]+sz[None,:]-inter
    J = np.where(un>0, inter/np.maximum(un,1e-9), 0.0)
    rel = R|R.T; iu = np.triu(np.ones_like(rel),1).astype(bool)
    big = (sz[:,None]>=5)&(sz[None,:]>=5)
    mi = iu & ~rel & big
    return float(J[mi].mean()) if mi.sum()>50 else float('nan')

S = 6
print(f'N=600, засевов {S}. ЦЕЛЬ: доля 0.10, разм. 4.00, '
      f'спектр +0.010 (с.к.о. 0.003), J несравн. 0.17')
print()
print(f"{'окно':>7}{'доб.':>6}{'доля':>9}{'разм.':>7}{'высота':>7}"
      f"{'<спектр>':>11}{'с.к.о.':>9}{'J несравн':>11}{'связ/соб':>9}")
for win in (0.02, 0.1, 0.3, 1.0):
    for extra in (0, 1, 3):
        v,f,d,h,l,jj = [],[],[],[],[],[]
        for t in range(S):
            rs = np.random.default_rng(91000+977*t)
            R = grow_one(600, win, extra, rs); ff = frac_pairs(R)
            v.append(spectral(R)); f.append(ff); d.append(mm_dim(ff))
            h.append(height(R)[0]); l.append(links_of(R).sum()/600)
            jj.append(jprof(R)); del R
        v = np.array(v)
        print(f'{win:>7.2f}{extra:>6}{np.mean(f):>9.4f}{np.mean(d):>7.2f}'
              f'{np.mean(h):>7.1f}{v.mean():>+11.4f}{v.std():>9.4f}'
              f'{np.nanmean(jj):>11.4f}{np.mean(l):>9.1f}', flush=True)
