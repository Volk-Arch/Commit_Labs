"""M наклонных направлений, ЗАФИКСИРОВАННЫХ на весь мир.

Шум шёл от перерисовки направления c на каждое событие. Фиксируем M
направлений один раз на мир, на событие оставляем только M порогов:

    D = { b : (C v(b)^2)_m <= w_m ,  m = 1..M }

Это покомпонентное доминирование на M положительных комбинациях
КВАДРАТОВ собственных времён. Замкнутость вниз цела. Ящик — особый
случай C = единичная. Усреднение по M порогам то же, что у ящика, значит
воспроизводимость должна вернуться, а наклон остаться.
"""
import math, numpy as np
from core import (mm_dim, height, spectral, frac_pairs, links_of,
                  transitivity_violations)

def jinc(R):
    A=R.astype(np.float32); it=A@A.T; sz=A.sum(1)
    un=sz[:,None]+sz[None,:]-it
    J=np.where(un>0, it/np.maximum(un,1e-9), 0.0)
    rel=R|R.T; iu=np.triu(np.ones_like(rel),1).astype(bool)
    big=(sz[:,None]>=5)&(sz[None,:]>=5); m=iu&~rel&big
    return float(J[m].mean()) if m.sum()>50 else float('nan')

def grow(N, kref, M, s, rs, conc=0.5, sq=True):
    C = rs.dirichlet(np.full(kref, conc), M)      # ФИКСИРОВАНЫ на мир
    R=np.zeros((N,N),bool); V=np.zeros((N,kref),np.int32)
    for i in range(kref): V[i,i]=1
    for k in range(kref, N):
        Vk=V[:k].astype(np.float64)
        Q=(Vk**2 if sq else Vk) @ C.T                 # k x M
        mx=Q.max(0)+1.0
        w=rs.random(M)*s*mx
        d=(Q<=w[None,:]).all(1)
        R[k,:k]=d
        if d.any():
            sub=V[:k][d]
            for i in range(kref):
                col=sub[:,i]; pos=col[col>0]
                V[k,i]=(pos.max()+1) if len(pos) else 0
    return R

S=8
print('ЦЕЛЬ конуса: доля 0.098, разм. 4.03, спектр +0.0145@800 / '
      '+0.0213@1600,')
print('с.к.о. 0.0026/0.0016, высота 10.2@1200, связей ~18@1000, J 0.169')
print('(ящик на v: спектр +0.0019 +-0.0034, доля 0.0948)')
print()
print(f"{'k':>4}{'M':>3}{'s':>6}{'conc':>6}{'N':>6}{'доля':>9}{'разм.':>7}"
      f"{'высота':>7}{'<спектр>':>11}{'с.к.о.':>9}{'наруш':>6}{'связ':>7}"
      f"{'J':>8}")
for kr, M, sc, cc in ((6,6,1.0,0.5),(6,6,1.0,0.15),(6,4,1.0,0.5),
                      (6,8,1.0,0.5),(6,6,0.6,0.5),(6,6,1.5,0.5)):
    for N in (800, 1600):
        v,f,d,h,l,j,tv=[],[],[],[],[],[],[]
        for t in range(S):
            rs=np.random.default_rng(161000+977*t)
            Rm=grow(N,kr,M,sc,rs,cc); ff=frac_pairs(Rm)
            v.append(spectral(Rm)); f.append(ff); d.append(mm_dim(ff))
            h.append(height(Rm)[0]); l.append(links_of(Rm).sum()/N)
            j.append(jinc(Rm)); tv.append(transitivity_violations(Rm)); del Rm
        v=np.array(v)
        print(f'{kr:>4}{M:>3}{sc:>6.2f}{cc:>6.2f}{N:>6}{np.mean(f):>9.4f}'
              f'{np.nanmean(d):>7.2f}{np.mean(h):>7.1f}{v.mean():>+11.4f}'
              f'{v.std():>9.4f}{int(np.sum(tv)):>6}{np.mean(l):>7.1f}'
              f'{np.nanmean(j):>8.4f}', flush=True)
