"""Непрерывные эталоны атома 3+1 (тот же прибор: окно H, Ханн, вес e^{−λr} в шаре R) для m = 1/τ², α = 0.8, a = 0.5:
КГ 1s, наклон по массе, второй порядок (вставка z), спиновый сдвиг Дирака, J «гелия» (облако КГ 1s, e² = α/2)."""
import numpy as np, scipy.linalg as sl, scipy.optimize as so, json, sys
TAU=(1+5**.5)/2; m=1/TAU**2; al=float(sys.argv[3]) if len(sys.argv)>3 else 0.6; a=0.5; R=float(sys.argv[1]) if len(sys.argv)>1 else 11.0; H=float(sys.argv[2]) if len(sys.argv)>2 else 24.0
lam=min(al*m,0.7)                                   # вес/источник e^{−λr}: λ = 1/(боровский радиус)
h=0.04; Rb=60.0; r=h*np.arange(1,int(Rb/h)+1); N=len(r); prof=np.exp(-lam*r); ball=r<R
def lap(u,l):
    d=np.empty_like(u); d[1:-1]=(u[2:]-2*u[1:-1]+u[:-2])/h**2; d[0]=(u[1]-2*u[0])/h**2; d[-1]=(-2*u[-1]+u[-2])/h**2; return d-l*(l+1)/r**2*u
def evolve(U,m2,dS=None,fsp=None,dt=0.02):
    f=np.zeros(N,complex) if fsp is None else fsp
    y=np.array([np.zeros(N,complex),np.zeros(N,complex),(r*prof).astype(complex),np.zeros(N,complex)]); ts=[]; sig=[]; t=0.0
    def F(y):
        us,up,vs,vp=y
        return np.array([vs,vp,2j*U*vs+U*U*us+lap(us,0)-m2*us+f*up,2j*U*vp+U*U*up+lap(up,1)-m2*up+f*us])
    while t<H-0.2:
        k1=F(y);k2=F(y+dt/2*k1);k3=F(y+dt/2*k2);k4=F(y+dt*k3); y=y+dt/6*(k1+2*k2+2*k3+k4); t+=dt; ts.append(t); sig.append(np.sum((r*prof*y[0])[ball])*h)
    return np.array(ts),np.array(sig)
def peak(ts,sig,om=np.linspace(0.05,0.8,3001)):
    s=(ts>1.5)&(ts<H-0.3); tt=ts[s]; w=np.sin(np.pi*(tt-tt.min())/np.ptp(tt))**2
    S=np.abs(np.exp(-1j*np.outer(om,tt))@(w*sig[s]))**2; i=int(np.argmax(S)); A,B,C=S[i-1],S[i],S[i+1]; return om[i]+0.5*(A-C)/(A-2*B+C)*(om[1]-om[0])
U=-al/np.sqrt(r*r+a*a); dU=al*r/(r*r+a*a)**1.5
E1=peak(*evolve(U,m*m)); E2=peak(*evolve(U,0.81*m*m))
cz=0.12*m*m/(2/np.e); Es=peak(*evolve(U,m*m,fsp=cz*r*np.exp(-r/2)/np.sqrt(3)))
Ed=peak(*evolve(U,m*m,fsp=1j*dU))
# точные уровни (без прибора)
D=(np.diag(np.full(N,2.0))-np.diag(np.ones(N-1),1)-np.diag(np.ones(N-1),-1))/h**2
lamb=lambda om,Ut: sl.eigh(D+np.diag(m*m-(om-Ut)**2),eigvals_only=True,subset_by_index=[0,0])[0]
Ex=so.brentq(lambda o:lamb(o,U),0.05,m*0.99999)
_,v=sl.eigh(D+np.diag(m*m-(Ex-U)**2),subset_by_index=[0,0]); u=v[:,0]; sig=(Ex-U)*u*u; sig/=np.sum(sig)*h
Q=np.cumsum(sig)*h; out=np.cumsum((sig/r)[::-1])[::-1]*h; Uc=(al/2)*(Q/r+out); Jx=so.brentq(lambda o:lamb(o,U+Uc),0.05,m*0.99999)-Ex
EJ=peak(*evolve(U+Uc,m*m))
ref=dict(m=m,al=al,a=a,R=R,H=H,lam=lam,E=E1,slope=(E1-E2)/(0.19*m*m),stark=Es-E1,spin=Ed-E1,J=EJ-E1,E_exact=Ex,J_exact=Jx)
json.dump(ref,open('atom_fz_ref_R%g_H%g_a%g.json'%(R,H,al),'w')); np.savez('atom_fz_cloud_a%g.npz'%al,r=r,sig_r=sig/(4*np.pi*r*r),Uc=Uc)
print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in ref.items()},ensure_ascii=False))
