"""Слои, усреднённые по прореживаниям (масштаб нелокальности): средний мир плотности ερ, веса по числу событий n в интервале
w̃_n = Σ_k w_k C(n,k) ε^k (1−ε)^{n−k}, множитель c(ερ). Среднее ядро — то же, что у слоёв при ερ (хвост убит), а шум усреднён."""
import sys, math, time, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd, causal as Cs, propagate as Pr
H,Rc=8.0,9.0; al=0.45; rho=float(sys.argv[1]); kind=sys.argv[2]; eps=float(sys.argv[3]); nmax=int(sys.argv[4]); tcf=float(sys.argv[5])
if kind=='gold': X=Wd.gold31(rho,'cylinder',H=H,Rc=Rc,Rs=50.0).X
else:
    rng=np.random.default_rng(1); V=(H+0.7)*(2*Rc)**3; n=rng.poisson(rho*V); P=np.c_[rng.uniform(-0.6,H+0.1,n),rng.uniform(-Rc,Rc,(n,3))]
    X=P[np.linalg.norm(P[:,1:],axis=1)<=Rc]; X=X[np.argsort(X[:,0])]
T0=time.time(); I,J,N=Cs.layers31(X,rho,nmax,tcf); n=len(X); print('%s ρ=%g ε=%g: событий %d, пар %d (%.0f с); макс n у пар: τ-срез ок? доля n=nmax %.3f'%(kind,rho,eps,n,len(I),time.time()-T0,(N==nmax).mean()),flush=True)
r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; f=np.where(t<0.45,np.exp(-al*r),0); cen=r<2.0
cs=math.sqrt(eps)   # c(ερ)/c(ρ)
for nm,w in (('связи',(1.0,)),('2 слоя',(2.0,-2.0)),('3 слоя',(3.0,-7.0,4.0))):
    wt=np.array([sum(wk*math.comb(nn,k)*eps**k*(1-eps)**(nn-k) for k,wk in enumerate(w) if k<=nn) for nn in range(nmax+1)])*cs
    Wt=wt[N.astype(int)]; out=[]
    for mm in (0.25,0.5,1.0,2.0):
        phi=Pr.johnston31_respond(n,I,J,rho,math.sqrt(mm),f,Wt)
        ed=np.arange(0,H+1e-9,0.5); k_=np.digitize(t,ed)-1; ok=(k_>=0)&(k_<len(ed)-1)&cen
        amp=np.sqrt(np.bincount(k_[ok],np.abs(phi)[ok]**2,minlength=len(ed)-1)/np.maximum(np.bincount(k_[ok],minlength=len(ed)-1),1))
        cnt=np.bincount(k_[ok],minlength=len(ed)-1); tc=0.5*(ed[1:]+ed[:-1]); s=(tc>2)&(tc<7)&(cnt>0); out.append(np.polyfit(tc[s],np.log(amp[s]+1e-300),1)[0])
    print('   %s (ε=%g): рост при m² = 0.25, 0.5, 1, 2: %s'%(nm,eps,np.round(out,3)),flush=True)
