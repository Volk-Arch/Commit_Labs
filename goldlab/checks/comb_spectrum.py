"""Спектр «гребёнки»: ядро размытого конуса на кристалле при k = 0 как функция Δt, его преобразование Фурье против непрерывного.
Отклонение D(ω) = Σ c′w e^{iωΔt} − ∫ t e^{iωt} dt показывает брэгговские частоты зерна."""
import sys, math, numpy as np, numba
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd
from goldlab.core import r_for_density
from fuzzy_calib import acc
rho=float(sys.argv[1]); beta=0.115; S=float(sys.argv[2]) if len(sys.argv)>2 else 9.0; T=6.0; c=math.sqrt(rho)/(2*math.pi*math.sqrt(6)); db=0.005
bins=np.arange(0,T+1e-9,db); out={}
for kind in ('gold','poisson'):
    tot=np.zeros(len(bins)-1); ntg=0; rng=np.random.default_rng(11)
    for k in range(3):
        if kind=='gold':
            v=rng.normal(size=4); v*=r_for_density(rho,4)*rng.random()**0.25/np.linalg.norm(v)
            G=Wd.points_in_box31([(-T-0.5,0.6),(-S,S),(-S,S),(-S,S)],Wd.r_for_density31(rho),tuple(v))
        else:
            n=rng.poisson(rho*(T+1.1)*(2*S)**3); G=np.c_[rng.uniform(-T-0.5,0.6,n),rng.uniform(-S,S,(n,3))]
        G=np.ascontiguousarray(G); cand=np.nonzero((np.abs(G[:,0])<0.5)&(np.linalg.norm(G[:,1:],axis=1)<2.5))[0]; tg=rng.choice(cand,min(80,len(cand)),replace=False)
        tot+=acc(G,tg,(2.8*rho**-0.25)**2,rho*math.pi/24,beta,bins); ntg+=len(tg)
    out[kind]=c*tot/ntg/db
tm=0.5*(bins[1:]+bins[:-1]); win=np.where(tm<5.5,1.0,np.cos(np.pi*(tm-5.5)/1.0)**2*(tm<6.5))   # мягкое окно на краю
om=np.linspace(0.05,12,2400)
for kind,cA in out.items():
    dev=(cA-tm)*win; D=np.abs(np.exp(1j*np.outer(om,tm))@dev*db)
    idx=[i for i in range(1,len(om)-1) if D[i]>D[i-1] and D[i]>D[i+1]]; idx=sorted(idx,key=lambda i:-D[i])[:8]
    print('%s ρ=%g: |D(ω)| пики (ω: высота): %s; |D| при ω=1, 2: %.3f, %.3f'%(kind,rho,' '.join('%.2f:%.2f'%(om[i],D[i]) for i in sorted(idx)),D[np.argmin(abs(om-1))],D[np.argmin(abs(om-2))]),flush=True)
    np.save('comb_%s_rho%g.npy'%(kind,rho),np.c_[om,D])
