"""Резонанс по массе: свободная частота покоя размытого конуса (β = 0.115, голая масса из ядра) против массы на кристалле и на посыпке,
и непрерывная теория тем же прибором. Особенности, которые есть у кристалла и нет у посыпки, — отклик на гребёнку."""
import sys, math, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd, observe as Ob, fuzzy as Fz
rho=float(sys.argv[1]); R=float(sys.argv[2]); H=12.0; al=0.45; beta=0.115
ms=np.array([0.2,0.35,0.5,0.65,0.8,0.9,1.0,1.1,1.25,1.4,1.6,1.8])
t_,d_=Fz.mean_kernel_delta(rho,beta); G=lambda w: np.trapezoid(np.exp(1j*w*t_)*d_,t_)
mb2=[m*m/(1-m*m*G(m).real) for m in ms]
def world(kind):
    if kind=='кристалл': return np.ascontiguousarray(Wd.gold31(rho,'cylinder',H=H,Rc=R,Rs=50.0).X)
    rng=np.random.default_rng(1); n=rng.poisson(rho*(H+0.7)*(2*R)**3); P=np.c_[rng.uniform(-0.6,H+0.1,n),rng.uniform(-R,R,(n,3))]
    P=P[np.linalg.norm(P[:,1:],axis=1)<=R]; return np.ascontiguousarray(P[np.argsort(P[:,0])])
res={}
for kind in ('кристалл','посыпка'):
    X=world(kind); n=len(X); r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; prof=np.exp(-al*r); sel=(t>1.5)&(t<H-0.3); e=np.array([1,0],complex)
    f=np.ascontiguousarray(np.where((t<0.45)[:,None],prof[:,None]*e[None,:],0).astype(np.complex128)); om=np.linspace(0.05,2.6,2551)
    alpha=rho*math.pi/24; coef=math.sqrt(rho)/(2*math.pi*math.sqrt(6)); tc2=(2.8*rho**-0.25)**2; out=[]
    for m2 in mb2:
        B=np.ascontiguousarray(np.broadcast_to(np.eye(2)*(-m2/rho),(n,2,2)),dtype=np.complex128)
        s=Fz.respond_fly(X,tc2,alpha,coef,0.0,0.5,Fz.GX,Fz.GW,f,B,False,beta)@e; S=Ob.spectrum(t[sel],prof[sel]*s[sel],om,'hann'); i=int(np.argmax(S*(om>0.12))); out.append(om[i])
    res[kind]=np.array(out); print(kind,np.round(res[kind],4),flush=True)
src=open('instr_cont.py').read().split('E0=peak')[0].replace('ball=r<7','ball=r<%g'%R).replace('T=19.8','T=11.8').replace('ts<19.7','ts<11.7').replace('om=np.linspace(0.6,1.3,1401)','om=np.linspace(0.05,2.6,2551)')
g={}; exec(src,g); g['U'][:]=0; cont=np.array([g['peak'](*g['evolve'](m2=m*m)) for m in ms]); print('непрерывная',np.round(cont,4))
np.savez('mass_scan_rho%g_R%g.npz'%(rho,R),ms=ms,cr=res['кристалл'],po=res['посыпка'],co=cont)
for i,m in enumerate(ms): print('m=%.2f: кристалл %.4f, посыпка %.4f, непрерывная %.4f; кристалл − посыпка %+.4f'%(m,res['кристалл'][i],res['посыпка'][i],cont[i],res['кристалл'][i]-res['посыпка'][i]))
