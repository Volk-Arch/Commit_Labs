"""Слои без краёв: цилиндр R = 9, окно 8, без конуса источника. У событий в центре (r < 2) весь прошлый конус внутри мира ⇒
число событий в интервале точное. Рост меряем по центру на t = 2…7."""
import sys, math, time, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd, causal as Cs, propagate as Pr
H,Rc=8.0,9.0; al=0.45; rho=float(sys.argv[1]); kind=sys.argv[2]
if kind=='gold': X=Wd.gold31(rho,'cylinder',H=H,Rc=Rc,Rs=50.0).X
else:
    rng=np.random.default_rng(1); V=(H+0.7)*(2*Rc)**3; n=rng.poisson(rho*V); P=np.c_[rng.uniform(-0.6,H+0.1,n),rng.uniform(-Rc,Rc,(n,3))]
    X=P[np.linalg.norm(P[:,1:],axis=1)<=Rc]; X=X[np.argsort(X[:,0])]
T0=time.time(); I,J,N=Cs.layers31(X,rho,2,3.0); n=len(X); print('%s ρ=%g: событий %d, пар %d (%.0f с)'%(kind,rho,n,len(I),time.time()-T0),flush=True)
r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; prof=np.exp(-al*r); f=np.where(t<0.45,prof,0); cen=r<2.0
for nm,w in (('связи',(1.0,)),('2 слоя',(2.0,-2.0)),('3 слоя',(3.0,-7.0,4.0))):
    Wt=np.zeros(len(I))
    for k,wk in enumerate(w): Wt[N==k]=wk
    keep=Wt!=0; out=[]
    for mm in (0.25,0.5,1.0,2.0):
        phi=Pr.johnston31_respond(n,I[keep],J[keep],rho,math.sqrt(mm),f,Wt[keep])
        ed=np.arange(0,H+1e-9,0.5); k_=np.digitize(t,ed)-1; ok=(k_>=0)&(k_<len(ed)-1)&cen
        amp=np.sqrt(np.bincount(k_[ok],np.abs(phi)[ok]**2,minlength=len(ed)-1)/np.maximum(np.bincount(k_[ok],minlength=len(ed)-1),1))
        cnt=np.bincount(k_[ok],minlength=len(ed)-1); tc=0.5*(ed[1:]+ed[:-1]); s=(tc>2)&(tc<7)&(cnt>0); out.append(np.polyfit(tc[s],np.log(amp[s]+1e-300),1)[0])
    print('   %s: рост при m² = 0.25, 0.5, 1, 2: %s'%(nm,np.round(out,3)),flush=True)
