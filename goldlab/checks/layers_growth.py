"""Разгон против слоёв: связи (1), два слоя (2, −2), три слоя (3, −7, 4). Кристалл и посыпка, цилиндр H=12, R=5."""
import sys, math, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd, causal as Cs, propagate as Pr
from growth_poisson import cyl_poisson
H=12.0; al,a=0.45,0.5
def run(X,rho,M2):
    I,J,N=Cs.layers31(X,rho,2,3.0); n=len(X); r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; prof=np.exp(-al*r); f=np.where(t<0.45,prof,0)
    res={}
    for nm,w in (('связи',(1.0,)),('2 слоя',(2.0,-2.0)),('3 слоя',(3.0,-7.0,4.0))):
        W=np.zeros(len(I)); 
        for k,wk in enumerate(w): W[N==k]=wk
        keep=W!=0; out=[]
        for mm in M2:
            phi=Pr.johnston31_respond(n,I[keep],J[keep],rho,math.sqrt(mm),f,W[keep])
            ed=np.arange(0,H+1e-9,1.0); k_=np.digitize(t,ed)-1; ok=(k_>=0)&(k_<len(ed)-1)
            amp=np.sqrt(np.bincount(k_[ok],np.abs(prof*phi)[ok]**2,minlength=len(ed)-1)/np.maximum(np.bincount(k_[ok],minlength=len(ed)-1),1))
            tc=0.5*(ed[1:]+ed[:-1]); s=(tc>2)&(tc<H-1); out.append(np.polyfit(tc[s],np.log(amp[s]+1e-300),1)[0])
        res[nm]=np.round(out,3)
    return res
M2=(0.25,1.0,2.0)
for rho in (1.0,2.0):
    print('кристалл ρ=%g, m²=%s: %s'%(rho,M2,run(Wd.gold31(rho,'cylinder',H=H,Rc=5,Rs=3.0).X,rho,M2)),flush=True)
    print('посыпка  ρ=%g, m²=%s: %s'%(rho,M2,run(cyl_poisson(rho,1),rho,M2)),flush=True)
