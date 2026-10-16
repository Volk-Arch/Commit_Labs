"""Калибровка размытого конуса по статистике САМОГО кристалла: накопленная сумма C(T) = ∫₀^T c′A(t)dt против T²/2.
Подгонка C(T) = κ·T²/2 + D на T ∈ [2, 6]: κ — нормировка (должна быть 1), D = ∫(c′A − t)dt — сдвиг массы (1/ω² = 1/m² + D)."""
import sys, math, numpy as np, numba
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd
from goldlab.core import r_for_density
@numba.njit
def acc(G, tg, tc2, alpha, beta, bins):
    out=np.zeros(len(bins)-1); db=bins[1]-bins[0]
    for j in tg:
        for i in range(G.shape[0]):
            dt=G[j,0]-G[i,0]
            if dt<=0 or dt>=bins[-1]: continue
            u=dt*dt-(G[j,1]-G[i,1])**2-(G[j,2]-G[i,2])**2-(G[j,3]-G[i,3])**2
            if u>=tc2 or u<=-tc2: continue
            out[int(dt/db)]+=0.5*math.exp(-alpha*u*u)*((1.0-beta) if u>0 else (1.0+beta))
    return out
if __name__=='__main__':
    rho=float(sys.argv[1]); kind=sys.argv[2]; beta=float(sys.argv[3]); S=9.0; T=6.0; c=math.sqrt(rho)/(2*math.pi*math.sqrt(6))
    bins=np.arange(0,T+1e-9,0.05); tot=np.zeros(len(bins)-1); ntg=0; rng=np.random.default_rng(11)
    for k in range(4):
        if kind=='gold':
            v=rng.normal(size=4); v*=r_for_density(rho,4)*rng.random()**0.25/np.linalg.norm(v)
            G=Wd.points_in_box31([(-T-0.5,0.6),(-S,S),(-S,S),(-S,S)],Wd.r_for_density31(rho),tuple(v))
        else:
            n=rng.poisson(rho*(T+1.1)*(2*S)**3); G=np.c_[rng.uniform(-T-0.5,0.6,n),rng.uniform(-S,S,(n,3))]
        G=np.ascontiguousarray(G); cand=np.nonzero((np.abs(G[:,0])<0.5)&(np.linalg.norm(G[:,1:],axis=1)<2.5))[0]; tg=rng.choice(cand,min(80,len(cand)),replace=False)
        tot+=acc(G,tg,(2.8*rho**-0.25)**2,rho*math.pi/24,beta,bins); ntg+=len(tg)
    cA=c*tot/ntg/0.05; tm=0.5*(bins[1:]+bins[:-1]); C=np.cumsum(cA)*0.05; s=(tm>2)&(tm<T-0.1)
    A=np.c_[tm[s]**2/2,np.ones(s.sum())]; (kap,D),*_=np.linalg.lstsq(A,C[s],rcond=None)
    print('%s ρ=%g β=%g (целей %d): κ = %.4f, D = %+.4f; C(T)−T²/2 при T=2,3,4,5,6: %s'%(kind,rho,beta,ntg,kap,D,' '.join('%+.3f'%(C[np.argmin(abs(tm-x))]-x*x/2) for x in (2,3,4,5,5.9))),flush=True)
