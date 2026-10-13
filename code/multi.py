"""M наклонных отсечений: D = {b : sum_i c_i^(m) v_i(b)^2 <= W_m, m=1..M}.

Ящик = M=k с осевыми c (ступенчатая граница, спектр подавлен).
Одно наклонное = M=1 (спектр верный, плотность взлетает).
Промежуточное: M наклонных. Порог берётся как КВАНТИЛЬ, чтобы каждое
отсечение оставляло известную долю, и плотность была под контролем.
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

def grow(N, kref, M, rs, conc=0.5):
    R=np.zeros((N,N),bool); V=np.zeros((N,kref),np.int32)
    for i in range(kref): V[i,i]=1
    for k in range(kref, N):
        Q=(V[:k].astype(np.float64))**2
        d=np.ones(k,bool)
        for _ in range(M):
            c=rs.dirichlet(np.full(kref, conc))
            q=Q@c
            W=rs.random()*(q.max()+1.0)
            d &= q<=W
        R[k,:k]=d
        if d.any():
            sub=V[:k][d]
            for i in range(kref):
                col=sub[:,i]; pos=col[col>0]
                V[k,i]=(pos.max()+1) if len(pos) else 0
    return R

S=5
print('ЦЕЛЬ конуса: доля 0.098, разм. 4.03, спектр +0.0145@800 / '
      '+0.0213@1600, J 0.169')
print('высота 10.2@1200, связей ~18@1000')
print()
print(f"{'k':>4}{'M':>4}{'conc':>6}{'N':>6}{'доля':>9}{'разм.':>7}"
      f"{'высота':>7}{'<спектр>':>11}{'с.к.о.':>9}{'наруш':>7}"
      f"{'связ':>7}{'J':>8}")
for kr, M, cc in ((6,2,0.5),(6,3,0.5),(6,4,0.5),(6,6,0.5),
                  (6,4,0.15),(6,4,2.0),(8,4,0.5)):
    for N in (800, 1600):
        v,f,d,h,l,j,tv=[],[],[],[],[],[],[]
        for t in range(S):
            rs=np.random.default_rng(131000+977*t)
            Rm=grow(N,kr,M,rs,cc); ff=frac_pairs(Rm)
            v.append(spectral(Rm)); f.append(ff); d.append(mm_dim(ff))
            h.append(height(Rm)[0]); l.append(links_of(Rm).sum()/N)
            j.append(jinc(Rm)); tv.append(transitivity_violations(Rm)); del Rm
        v=np.array(v)
        print(f'{kr:>4}{M:>4}{cc:>6.2f}{N:>6}{np.mean(f):>9.4f}'
              f'{np.nanmean(d):>7.2f}{np.mean(h):>7.1f}{v.mean():>+11.4f}'
              f'{v.std():>9.4f}{int(np.sum(tv)):>7}{np.mean(l):>7.1f}'
              f'{np.nanmean(j):>8.4f}', flush=True)
