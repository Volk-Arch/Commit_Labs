"""Веса слоёв из статистики САМОГО кристалла (а не посыпки): моменты A_n^(k) = ⟨Σ_{y: n_xy = n} τ_xy^{2k} e^{−Δt}⟩ по событиям
в глубине (ансамбль сдвигов фазона). Условия те же: Σ w_n A_n^(0) = A_0^(0) (нормировка как у связей), Σ w_n A_n^(1) = 0
(нет δ′(τ²) ⇒ нет логарифма), Σ w_n A_n^(2) = 0 (следующий порядок)."""
import sys, math, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd
from goldlab.core import r_for_density
rho=float(sys.argv[1]); kind=sys.argv[2]; nmax=4; S=7.0; T=6.0; tc2=(3.0*rho**-0.25)**2
A=np.zeros((3,nmax+1)); ntg=0; rng=np.random.default_rng(7)
for wk in range(6):
    if kind=='gold':
        v=rng.normal(size=4); v*=r_for_density(rho,4)*rng.random()**0.25/np.linalg.norm(v)
        G=Wd.points_in_box31([(-T-0.5,0.6),(-S,S),(-S,S),(-S,S)],Wd.r_for_density31(rho),tuple(v))
    else:
        V=(T+1.1)*(2*S)**3; n=rng.poisson(rho*V); G=np.c_[rng.uniform(-T-0.5,0.6,n),rng.uniform(-S,S,(n,3))]
    tg=np.nonzero((np.abs(G[:,0])<0.3)&(np.linalg.norm(G[:,1:],axis=1)<0.7))[0]; tg=rng.choice(tg,min(25,len(tg)),replace=False)
    for j in tg:
        d=G[j]-G; tau2=d[:,0]**2-(d[:,1:]**2).sum(1); c_=np.nonzero((d[:,0]>0)&(tau2>1e-12)&(tau2<tc2))[0]
        Q=G[c_]; e=Q[None,:,:]-Q[:,None,:]; nn=((e[...,0]>0)&(e[...,0]**2-(e[...,1:]**2).sum(-1)>1e-12)).sum(1)
        g=np.exp(-d[c_,0]); u=tau2[c_]*math.sqrt(rho)
        for k in range(3):
            for m in range(nmax+1): A[k,m]+=np.sum((u**k*g)[nn==m])
        ntg+=1
A/=ntg; print('%s ρ=%g: целей %d; A_n^(0) = %s'%(kind,rho,ntg,np.round(A[0],3)))
print('   A_n^(1) = %s;  A_n^(2) = %s'%(np.round(A[1],3),np.round(A[2],3)))
for L in (2,3):
    M=A[:,:L].copy(); rhs=np.array([A[0,0],0,0])[:L]; w=np.linalg.solve(M[:L],rhs); print('   %d слоя: веса %s'%(L,np.round(w,3)))
np.save('crystal_weights_%s_rho%g.npy'%(kind,rho),A)
