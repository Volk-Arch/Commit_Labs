"""Сравнение не по пику, а по всему временному ряду наблюдаемой S(t) = ∫ e^{−αMr}·φ d³x: дискретный мир против непрерывной теории
(с одним общим комплексным множителем — нормировка источника). Плюс спектр с длинным «затуханием» не нужен: видно, где расходятся."""
import sys, math, json, numpy as np; sys.path.insert(0,'/home/claude/goldlab')
from goldlab import atom31f as Af
Af.AL=float(sys.argv[1]); M=float(sys.argv[2]); R=float(sys.argv[3]); H=float(sys.argv[4]); rho=float(sys.argv[5]); kind=sys.argv[6]
# непрерывная: ряд
def cont_series(**kw):
    import types
    h=0.04; Rbox=max(4*R,40.0); r=h*np.arange(1,int(Rbox/h)+1); N=len(r); U=-Af.AL/np.sqrt(r*r+Af.A*Af.A); prof=np.exp(-Af.AL*M*r); ball=r<R
    def lap(u,l):
        d=np.empty_like(u); d[1:-1]=(u[2:]-2*u[1:-1]+u[:-2])/h**2; d[0]=(u[1]-2*u[0])/h**2; d[-1]=(-2*u[-1]+u[-2])/h**2; return d-l*(l+1)/r**2*u
    def F(y):
        us,vs=y; return np.array([vs,2j*U*vs+U*U*us+lap(us,0)-M*M*us])
    y=np.array([np.zeros(N,complex),(r*prof).astype(complex)]); ts=[]; sig=[]; t=0.0; dt=0.01
    while t<H-0.2:
        k1=F(y);k2=F(y+dt/2*k1);k3=F(y+dt/2*k2);k4=F(y+dt*k3); y=y+dt/6*(k1+2*k2+2*k3+k4); t+=dt; ts.append(t); sig.append(np.sum((r*prof*y[0])[ball])*h*4*np.pi)
    return np.array(ts),np.array(sig)
tc,sc=cont_series()
X=Af.make_world(rho,H,R,poisson_seed=1 if kind=='poisson' else None); at=Af.Atom(X,rho,M,H,R); phi=at.respond(at.B(M))[:,0]
ed=np.arange(0.5,H-0.2,0.5); k=np.digitize(at.t,ed)-1; ok=(k>=0)&(k<len(ed)-1); cnt=np.bincount(k[ok],minlength=len(ed)-1)
Sd=(np.bincount(k[ok],(at.prof*phi)[ok].real,minlength=len(ed)-1)+1j*np.bincount(k[ok],(at.prof*phi)[ok].imag,minlength=len(ed)-1))/(rho*0.5)
tm=0.5*(ed[1:]+ed[:-1]); Sc=np.interp(tm,tc,sc.real)+1j*np.interp(tm,tc,sc.imag)
s=(tm>1.5)&(tm<6); c=np.vdot(Sc[s],Sd[s])/np.vdot(Sc[s],Sc[s])         # общий множитель — по раннему участку
print('множитель %.4f%+.4fi'%(c.real,c.imag))
for a,b in ((1.5,4),(4,8),(8,12),(12,16),(16,19.5)):
    q=(tm>a)&(tm<b); rel=np.linalg.norm(Sd[q]-c*Sc[q])/np.linalg.norm(c*Sc[q]); dph=np.angle(np.vdot(c*Sc[q],Sd[q]))
    print('t %4.1f–%4.1f: |дискр − непр|/|непр| = %.3f, сдвиг фазы %+.3f рад, |дискр|/|непр| = %.3f'%(a,b,rel,dph,np.linalg.norm(Sd[q])/np.linalg.norm(c*Sc[q])))
np.savez('tscmp_%s_rho%g.npz'%(kind,rho),tm=tm,Sd=Sd,Sc=Sc,c=c)
