"""Временной ряд при вставке «потенциал при энергии E» (без фаз): дискретный мир против непрерывной теории (источник и динамика в R)."""
import sys, numpy as np; sys.path.insert(0,'/home/claude/goldlab')
from goldlab import atom31f as Af, fuzzy as Fz
Af.AL=0.6; M=0.381966; R=11.0; H=20.0; E=float(sys.argv[1]); kind=sys.argv[2]; pauli=float(sys.argv[3]) if len(sys.argv)>3 else 0.0
X=Af.make_world(1.0,H,R,poisson_seed=1 if kind=='poisson' else None); at=Af.Atom(X,1.0,M,H,R); U=-Af.AL/np.sqrt(at.r**2+Af.A**2)
b=-(at.mb2(M)+2*E*U-U*U); B=(b[:,None,None]*at.eye+pauli*1j*at.dU[:,None,None]*at.sr)
phi=Fz.respond_fast(X,Fz.build_index(X,2.0),at.tc2,at.alpha,at.coef,at.rg,at.Unuc,at.src(),np.ascontiguousarray(B,dtype=np.complex128),Af.BETA,1,use_phase=0)[:,0]
ed=np.arange(0.5,H-0.2,float(sys.argv[4]) if len(sys.argv)>4 else 0.5); k=np.digitize(at.t,ed)-1; ok=(k>=0)&(k<len(ed)-1)
cnt=np.bincount(k[ok],minlength=len(ed)-1); Sd=(np.bincount(k[ok],(at.prof*phi)[ok].real,minlength=len(ed)-1)+1j*np.bincount(k[ok],(at.prof*phi)[ok].imag,minlength=len(ed)-1))/np.maximum(cnt,1)*(4/3*np.pi*R**3); tm=0.5*(ed[1:]+ed[:-1])
h=0.04; r=h*np.arange(1,int(44/h)+1); N=len(r); th=(r<R).astype(float); Uc=-Af.AL/np.sqrt(r*r+Af.A**2)*th; prof=np.exp(-Af.AL*M*r); ball=r<R
m2=(M*M+2*E*Uc-Uc*Uc)*th; fp=pauli*1j*Af.AL*r/(r*r+Af.A**2)**1.5*th
def lap(u,l):
    d=np.empty_like(u); d[1:-1]=(u[2:]-2*u[1:-1]+u[:-2])/h**2; d[0]=(u[1]-2*u[0])/h**2; d[-1]=(-2*u[-1]+u[-2])/h**2; return d-l*(l+1)/r**2*u
def F(y):
    us,up,vs,vp=y; return np.array([vs,vp,lap(us,0)-m2*us+fp*up,lap(up,1)-m2*up+fp*us])
y=np.array([np.zeros(N,complex),np.zeros(N,complex),(r*prof*th).astype(complex),np.zeros(N,complex)]); ts=[]; sig=[]; t=0.0; dt=0.01
while t<H-0.2:
    k1=F(y);k2=F(y+dt/2*k1);k3=F(y+dt/2*k2);k4=F(y+dt*k3); y=y+dt/6*(k1+2*k2+2*k3+k4); t+=dt; ts.append(t); sig.append(np.sum((r*prof*y[0])[ball])*h*4*np.pi)
ts=np.array(ts); sig=np.array(sig); Sc=np.interp(tm,ts,sig.real)+1j*np.interp(tm,ts,sig.imag)
s=(tm>1.5)&(tm<6); c=np.vdot(Sc[s],Sd[s])/np.vdot(Sc[s],Sc[s]); out=[]
for a,bb in ((1.5,4),(4,8),(8,12),(12,16),(16,19.5)):
    q=(tm>a)&(tm<bb); out.append('%.2f/%+.2f/%.2f'%(np.linalg.norm(Sd[q]-c*Sc[q])/np.linalg.norm(c*Sc[q]),np.angle(np.vdot(c*Sc[q],Sd[q])),np.linalg.norm(Sd[q])/np.linalg.norm(c*Sc[q])))
print('%s, E=%.3f, Паули %g: множитель %.3f%+.3fi; отклонение/фаза/амплитуда: %s'%(kind,E,pauli,c.real,c.imag,' '.join(out)),flush=True)
