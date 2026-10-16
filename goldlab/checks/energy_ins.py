"""Потенциал как вставка при энергии E: B = −(m_b² + 2E·U − U²)/ρ (+ член Паули), без фаз; самосогласование E = ω(E).
Эталон — непрерывная теория тем же прибором: источник и динамика только внутри области (как в дискретном мире)."""
import sys, json, numpy as np; sys.path.insert(0,'/home/claude/goldlab')
from goldlab import atom31f as Af, fuzzy as Fz
Af.AL=float(sys.argv[1]); M=float(sys.argv[2]); R=float(sys.argv[3]); H=float(sys.argv[4]); kind=sys.argv[5]; rho=1.0
# --- непрерывная (источник и динамика внутри R)
def cont(pauli=0.0):
    h=0.04; r=h*np.arange(1,int(max(4*R,40)/h)+1); N=len(r); th=(r<R).astype(float); U=-Af.AL/np.sqrt(r*r+Af.A**2)*th; prof=np.exp(-Af.AL*M*r); ball=r<R
    dU=Af.AL*r/(r*r+Af.A**2)**1.5*th; fp=pauli*1j*dU; m2=M*M*th
    def lap(u,l):
        d=np.empty_like(u); d[1:-1]=(u[2:]-2*u[1:-1]+u[:-2])/h**2; d[0]=(u[1]-2*u[0])/h**2; d[-1]=(-2*u[-1]+u[-2])/h**2; return d-l*(l+1)/r**2*u
    def F(y):
        us,up,vs,vp=y
        return np.array([vs,vp,2j*U*vs+U*U*us+lap(us,0)-m2*us+fp*up,2j*U*vp+U*U*up+lap(up,1)-m2*up+fp*us])
    y=np.array([np.zeros(N,complex),np.zeros(N,complex),(r*prof*th).astype(complex),np.zeros(N,complex)]); ts=[]; sig=[]; t=0.0; dt=0.01
    while t<H-0.2:
        k1=F(y);k2=F(y+dt/2*k1);k3=F(y+dt/2*k2);k4=F(y+dt*k3); y=y+dt/6*(k1+2*k2+2*k3+k4); t+=dt; ts.append(t); sig.append(np.sum((r*prof*y[0])[ball])*h)
    ts=np.array(ts); sig=np.array(sig); s=(ts>1.5)&(ts<H-0.3); tt=ts[s]; w=np.sin(np.pi*(tt-tt.min())/np.ptp(tt))**2
    om=np.linspace(0.15,1.3,2301); S=np.abs(np.exp(-1j*np.outer(om,tt))@(w*sig[s]))**2; i=int(np.argmax(S)); a,b,c=S[i-1],S[i],S[i+1]
    return om[i]+0.5*(a-c)/(a-2*b+c)*(om[1]-om[0])
if kind=='ref':
    E=cont(); Ed=cont(1.0); print(json.dumps(dict(ref_E=E,ref_spin=Ed-E))); sys.exit()
X=Af.make_world(rho,H,R,poisson_seed=1 if kind=='poisson' else None); at=Af.Atom(X,rho,M,H,R); e=np.array([1,0])
U=-Af.AL/np.sqrt(at.r**2+Af.A**2); f=at.src()
def omega(E,pauli=0.0):
    b=-(at.mb2(M)+2*E*U-U*U); B=(b[:,None,None]*at.eye+pauli*1j*at.dU[:,None,None]*at.sr)/rho
    phi=Fz.respond_fast(X,at.idx if hasattr(at,'idx') else Fz.build_index(X,2.0),at.tc2,at.alpha,at.coef,at.rg,at.Unuc,f,np.ascontiguousarray(B,dtype=np.complex128),Af.BETA,1,use_phase=0)@e
    return at.peak(phi)
at.idx=Fz.build_index(X,2.0)
def fixed(pauli=0.0,E0=None):
    E0=E0 or 0.8*M; w0=omega(E0,pauli); E1=w0; w1=omega(E1,pauli)
    for _ in range(4):
        if abs(w1-E1)<2e-4: break
        E2=E1-(w1-E1)*(E1-E0)/((w1-E1)-(w0-E0)); E0,w0,E1=E1,w1,E2; w1=omega(E1,pauli)
    return E1,w1
E,wE=fixed(); Ed,wd=fixed(1.0,E)
print(json.dumps(dict(kind=kind,E=E,check=wE,spin=Ed-E)),flush=True)
