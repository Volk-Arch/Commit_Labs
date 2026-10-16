import sys, math, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd, observe as Ob, fuzzy as Fz
rho=float(sys.argv[1]); R=float(sys.argv[2]); kind=sys.argv[3]; H=12.0; al=0.45; beta=0.115
ms=[float(x) for x in sys.argv[4].split(',')]
t_,d_=Fz.mean_kernel_delta(rho,beta); G=lambda w: np.trapezoid(np.exp(1j*w*t_)*d_,t_)
if kind=='кристалл': X=np.ascontiguousarray(Wd.gold31(rho,'cylinder',H=H,Rc=R,Rs=50.0).X)
else:
    rng=np.random.default_rng(1); n=rng.poisson(rho*(H+0.7)*(2*R)**3); P=np.c_[rng.uniform(-0.6,H+0.1,n),rng.uniform(-R,R,(n,3))]
    P=P[np.linalg.norm(P[:,1:],axis=1)<=R]; X=np.ascontiguousarray(P[np.argsort(P[:,0])])
n=len(X); r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; prof=np.exp(-al*r); sel=(t>1.5)&(t<H-0.3); e=np.array([1,0],complex)
f=np.ascontiguousarray(np.where((t<0.45)[:,None],prof[:,None]*e[None,:],0).astype(np.complex128)); om=np.linspace(0.1,6.0,2951)
alpha=rho*math.pi/24; coef=math.sqrt(rho)/(2*math.pi*math.sqrt(6)); tc2=(2.8*rho**-0.25)**2
for m in ms:
    m2=m*m/(1-m*m*G(m).real); B=np.ascontiguousarray(np.broadcast_to(np.eye(2)*(-m2/rho),(n,2,2)),dtype=np.complex128)
    s=Fz.respond_fly(X,tc2,alpha,coef,0.0,0.5,Fz.GX,Fz.GW,f,B,False,beta)@e; S=Ob.spectrum(t[sel],prof[sel]*s[sel],om,'hann')
    pk=sorted(Ob.peaks(om,S,0.05),key=lambda q:-q[1])[:4]
    ed=np.arange(0,H+1e-9,1.0); kk=np.digitize(t,ed)-1; ok=(kk>=0)&(kk<len(ed)-1); cnt=np.bincount(kk[ok],minlength=len(ed)-1)
    amp=np.sqrt(np.bincount(kk[ok],np.abs(prof*s)[ok]**2,minlength=len(ed)-1)/np.maximum(cnt,1)); tc_=0.5*(ed[1:]+ed[:-1]); ss=(tc_>2)&(tc_<H-1)&(cnt>0)
    print('%s ρ=%g R=%g m=%.2f (m_b²=%.3f): рост %+.3f; пики %s'%(kind,rho,R,m,m2,np.polyfit(tc_[ss],np.log(amp[ss]+1e-300),1)[0],' '.join('%.3f(%.2f)'%q for q in pk)),flush=True)
