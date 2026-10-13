"""Фиксированная ФОРМА прошлого, варьируется только положение.

У конуса прошлое каждого события одной формы и различается лишь сдвигом.
У нас форма была случайна на событие — отсюда шум при малой плотности.

Берём выведенную формулу интервала с ФИКСИРОВАННОЙ G правильной
сигнатуры (G = A^T eta A гарантирует 1+/3-, измерять её не нужно):

    u_i(p) = (v_1(p)^2 + G_ii - v_i(p)^2)/2
    I(p,q) = v_1(p)^2 + v_1(q)^2 - 2 u(p)^T G^{-1} u(q)
    D = замыкание_вниз{ b : I(w, v(b)) >= 0 и v(b) <= w }

Замыкание вниз берётся явно, поэтому транзитивность цела всегда.
k = 5 опор: G это 4x4 со сигнатурой 1+/3-, как для 4d Минковского.
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

def grow(N, rs, s=1.0, spread=1.0, kref=5):
    eta = np.diag([1.0]+[-1.0]*(kref-2))
    A = rs.normal(size=(kref-1, kref-1))
    G = A.T @ eta @ A
    Gi = np.linalg.inv(G)
    gii = np.diag(G).copy()
    R=np.zeros((N,N),bool); V=np.zeros((N,kref),np.int32)
    for i in range(kref): V[i,i]=1
    for k in range(kref, N):
        Vk=V[:k].astype(np.float64)
        t1=Vk[:,0]**2
        U=0.5*(t1[:,None] + gii[None,:] - Vk[:,1:]**2)      # k x (kref-1)
        mx=Vk.max(0)+1.0
        w=rs.random(kref)*spread*mx
        tw=w[0]**2
        uw=0.5*(tw + gii - w[1:]**2)
        I = tw + t1 - 2.0*(U @ (Gi @ uw))
        d = (I >= 0) & (Vk <= w[None,:]*s).all(1)
        if d.any():                                  # замыкание вниз
            Ak=R[:k,:k].astype(np.float32)
            d = d | ((Ak.T @ d.astype(np.float32)) > 0)
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
print()
print(f"{'s':>6}{'spread':>8}{'N':>6}{'доля':>9}{'разм.':>7}{'высота':>7}"
      f"{'<спектр>':>11}{'с.к.о.':>9}{'наруш':>7}{'связ':>7}{'J':>8}")
for sc, sp in ((1.0,1.0),(1.0,0.6),(0.7,1.0),(1.0,1.5),(0.5,1.0)):
    for N in (800, 1600):
        v,f,d,h,l,j,tv=[],[],[],[],[],[],[]
        for t in range(S):
            rs=np.random.default_rng(171000+977*t)
            Rm=grow(N,rs,sc,sp); ff=frac_pairs(Rm)
            v.append(spectral(Rm)); f.append(ff); d.append(mm_dim(ff))
            h.append(height(Rm)[0]); l.append(links_of(Rm).sum()/N)
            j.append(jinc(Rm)); tv.append(transitivity_violations(Rm)); del Rm
        v=np.array(v)
        print(f'{sc:>6.2f}{sp:>8.2f}{N:>6}{np.mean(f):>9.4f}'
              f'{np.nanmean(d):>7.2f}{np.mean(h):>7.1f}{v.mean():>+11.4f}'
              f'{v.std():>9.4f}{int(np.sum(tv)):>7}{np.mean(l):>7.1f}'
              f'{np.nanmean(j):>8.4f}', flush=True)
