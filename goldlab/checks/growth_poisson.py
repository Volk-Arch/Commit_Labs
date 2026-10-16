"""Разгон: кристалл против случайной посыпки той же плотности (тот же цилиндр, тот же пропагатор). Если посыпка устойчива —
виновата регулярность кристалла (резонанс), если нет — сама схема суммы по цепочкам."""
import sys, math, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd, causal as Cs, propagate as Pr
H,Rc,Rs=12.0,5.0,3.0; al,a=0.45,0.5
def cyl_poisson(rho,seed):
    rng=np.random.default_rng(seed); V=(H+0.7)*(2*Rc)**3; n=rng.poisson(rho*V); P=np.c_[rng.uniform(-0.6,H+0.1,n),rng.uniform(-Rc,Rc,(n,3))]
    rr=np.linalg.norm(P[:,1:],axis=1); P=P[rr<=np.minimum(Rc,Rs+np.maximum(P[:,0],0)+0.6)]; return P[np.argsort(P[:,0])]
def growth(X,rho,m2):
    I,J=Cs.links_sparse(X,rho,1.6); n=len(X); r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; prof=np.exp(-al*r); f=np.where(t<0.45,prof,0)
    out=[]
    for mm in m2:
        phi=Pr.johnston31_respond(n,I,J,rho,math.sqrt(mm),f,None)
        ed=np.arange(0,H+1e-9,1.0); k=np.digitize(t,ed)-1; ok=(k>=0)&(k<len(ed)-1)
        amp=np.sqrt(np.bincount(k[ok],np.abs(prof*phi)[ok]**2,minlength=len(ed)-1)/np.maximum(np.bincount(k[ok],minlength=len(ed)-1),1))
        tc=0.5*(ed[1:]+ed[:-1]); s=(tc>2)&(tc<H-1); out.append(np.polyfit(tc[s],np.log(amp[s]+1e-300),1)[0])
    return out, len(I)/n
M2=(0.25,0.5,1.0,2.0)
for rho in (1.0,2.0):
    g,l=growth(Wd.gold31(rho,'cylinder',H=H,Rc=Rc,Rs=Rs).X,rho,M2); print('кристалл ρ=%g (связей %.0f): рост при m² = %s: %s'%(rho,l,M2,np.round(g,3)),flush=True)
    for sd in (1,2):
        g,l=growth(cyl_poisson(rho,sd),rho,M2); print('посыпка  ρ=%g #%d (связей %.0f): %s'%(rho,sd,l,np.round(g,3)),flush=True)
