"""Временной ряд дискретного мира против непрерывной теории: с ядром и без, с массой и без — где расходится?"""
import sys, numpy as np; sys.path.insert(0,'/home/claude/goldlab')
from goldlab import atom31f as Af
V=float(sys.argv[1]); M=float(sys.argv[2]); TR=int(sys.argv[3]) if len(sys.argv)>3 else 0; R=11.0; H=20.0; Af.AL=0.6; Af.A=0.5
X=Af.make_world(1.0,H,R,poisson_seed=1); at=Af.Atom(X,1.0,max(M,1e-6),H,R)
at.prof=np.exp(-0.6*0.381966*at.r)                         # форма источника и чтения — как у атома
# ядро: V·(−1/√(r²+a²)); голая масса из ядра (M=0 — без массы)
f=np.ascontiguousarray(np.where((at.t<0.45)[:,None],at.prof[:,None]*np.eye(2)[0][None,:],0).astype(np.complex128))
B=(-(at.mb2(M) if M>0 else 0.0))*at.eye/1.0
from goldlab import fuzzy as Fz
phi=Fz.respond_fast(X,at.idx if hasattr(at,'idx') else Fz.build_index(X,2.0),at.tc2,at.alpha,at.coef,np.array([V,0.5]),at.Unuc,f,np.ascontiguousarray(B,dtype=np.complex128),Af.BETA,1,use_phase=2 if V>0 else 0)[:,0]
ed=np.arange(0.5,H-0.2,0.5); k=np.digitize(at.t,ed)-1; ok=(k>=0)&(k<len(ed)-1)
Sd=(np.bincount(k[ok],(at.prof*phi)[ok].real,minlength=len(ed)-1)+1j*np.bincount(k[ok],(at.prof*phi)[ok].imag,minlength=len(ed)-1))/0.5; tm=0.5*(ed[1:]+ed[:-1])
h=0.04; r=h*np.arange(1,int(44/h)+1); N=len(r); th=(r<R).astype(float); U=-V/np.sqrt(r*r+0.25)*(th if TR else 1); prof=np.exp(-0.6*0.381966*r); ball=r<R; srcp=prof*th; MM=M*M*(th if TR else 1)
def lap(u):
    d=np.empty_like(u); d[1:-1]=(u[2:]-2*u[1:-1]+u[:-2])/h**2; d[0]=(u[1]-2*u[0])/h**2; d[-1]=(-2*u[-1]+u[-2])/h**2; return d
def F(y):
    us,vs=y; return np.array([vs,2j*U*vs+U*U*us+lap(us)-MM*us])
y=np.array([np.zeros(N,complex),(r*srcp).astype(complex)]); ts=[]; sig=[]; t=0.0; dt=0.01
while t<H-0.2:
    k1=F(y);k2=F(y+dt/2*k1);k3=F(y+dt/2*k2);k4=F(y+dt*k3); y=y+dt/6*(k1+2*k2+2*k3+k4); t+=dt; ts.append(t); sig.append(np.sum((r*prof*y[0])[ball])*h*4*np.pi)
ts=np.array(ts); sig=np.array(sig); Sc=np.interp(tm,ts,sig.real)+1j*np.interp(tm,ts,sig.imag)
s=(tm>1.5)&(tm<6); c=np.vdot(Sc[s],Sd[s])/np.vdot(Sc[s],Sc[s]); out=[]
for a,b in ((1.5,4),(4,8),(8,12),(12,16),(16,19.5)):
    q=(tm>a)&(tm<b); out.append('%.2f/%+.2f/%.2f'%(np.linalg.norm(Sd[q]-c*Sc[q])/np.linalg.norm(c*Sc[q]),np.angle(np.vdot(c*Sc[q],Sd[q])),np.linalg.norm(Sd[q])/np.linalg.norm(c*Sc[q])))
print('[источник внутри R%s] ядро V=%g, масса M=%g: множитель %.3f%+.3fi; отклонение/фаза/амплитуда: %s'%(', динамика обрезана' if TR else '',V,M,c.real,c.imag,' '.join(out)),flush=True)
