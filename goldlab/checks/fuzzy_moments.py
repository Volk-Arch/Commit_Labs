"""Средняя связь «размытого конуса» на настоящем мире: c′·Σ_y w(u_xy) по слоям Δt, против непрерывного t. Хвост t·δ(t) — есть ли δ′
(первый момент) на кристалле/посыпке, хотя в теории он нулевой?"""
import sys, math, numpy as np, numba
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd
@numba.njit
def acc(G, tg, tc2, alpha, bins, lam_shape):
    out=np.zeros(len(bins)-1); db=bins[1]-bins[0]
    for j in tg:
        for i in range(G.shape[0]):
            dt=G[j,0]-G[i,0]
            if dt<=0 or dt>=bins[-1]: continue
            u=dt*dt-(G[j,1]-G[i,1])**2-(G[j,2]-G[i,2])**2-(G[j,3]-G[i,3])**2
            if u>=tc2 or u<=-tc2: continue
            out[int(dt/db)]+=0.5*math.exp(-alpha*u*u)
    return out
rho=1.0; S=9.0; T=8.0
for kind in ('gold','poisson'):
    if kind=='gold': G=Wd.points_in_box31([(-T-0.5,0.6),(-S,S),(-S,S),(-S,S)],Wd.r_for_density31(rho))
    else:
        rng=np.random.default_rng(5); n=rng.poisson(rho*(T+1.1)*(2*S)**3); G=np.c_[rng.uniform(-T-0.5,0.6,n),rng.uniform(-S,S,(n,3))]
    G=np.ascontiguousarray(G); rng=np.random.default_rng(2)
    cand=np.nonzero((np.abs(G[:,0])<0.5)&(np.linalg.norm(G[:,1:],axis=1)<3.0))[0]; tg=rng.choice(cand,min(150,len(cand)),replace=False)
    bins=np.arange(0,6.0+1e-9,0.5); c=math.sqrt(rho)/(2*math.pi*math.sqrt(6))
    A=acc(G,tg,(2.8*rho**-0.25)**2,rho*math.pi/24,bins,1)/len(tg)/0.5; tm=0.5*(bins[1:]+bins[:-1]); cA=c*A
    print('%s ρ=%g (целей %d): c′A(t)/t: %s'%(kind,rho,len(tg),' '.join('%.0f:%.3f'%(a,b) for a,b in zip(tm[1::2],(cA/tm)[1::2]))))
    print('    ∫(c′A − t)dt на 0…6 = %+.3f; среднее t·δ на t = 3…6: %+.3f'%(np.sum(cA-tm)*0.5,np.mean((tm*(cA-tm))[6:])),flush=True)
