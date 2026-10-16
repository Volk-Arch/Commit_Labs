import numpy as np, sys
AL=0.6; A=0.5; M=0.381966; R=11.0; H=20.0
h=0.04; r=h*np.arange(1,int(44/h)+1); N=len(r); th=(r<R).astype(float); U=-AL/np.sqrt(r*r+A*A)*th; prof=np.exp(-AL*M*r); ball=r<R
def lap(u):
    d=np.empty_like(u); d[1:-1]=(u[2:]-2*u[1:-1]+u[:-2])/h**2; d[0]=(u[1]-2*u[0])/h**2; d[-1]=(-2*u[-1]+u[-2])/h**2; return d
def omega(E,pauli=0.0):
    m2=(M*M+2*E*U-U*U)*th; fp=pauli*1j*AL*r/(r*r+A*A)**1.5*th
    def lap1(u): return lap(u)-2/r**2*u
    def F(y):
        us,up,vs,vp=y; return np.array([vs,vp,lap(us)-m2*us+fp*up,lap1(up)-m2*up+fp*us])
    y=np.array([np.zeros(N,complex),np.zeros(N,complex),(r*prof*th).astype(complex),np.zeros(N,complex)]); ts=[]; sig=[]; t=0.0; dt=0.01
    while t<H-0.2:
        k1=F(y);k2=F(y+dt/2*k1);k3=F(y+dt/2*k2);k4=F(y+dt*k3); y=y+dt/6*(k1+2*k2+2*k3+k4); t+=dt; ts.append(t); sig.append(np.sum((r*prof*y[0])[ball])*h)
    ts=np.array(ts); sig=np.array(sig); s=(ts>1.5)&(ts<H-0.3); tt=ts[s]; w=np.sin(np.pi*(tt-tt.min())/np.ptp(tt))**2
    om=np.linspace(0.15,1.3,2301); S=np.abs(np.exp(-1j*np.outer(om,tt))@(w*sig[s]))**2; i=int(np.argmax(S)); a,b,c=S[i-1],S[i],S[i+1]
    return om[i]+0.5*(a-c)/(a-2*b+c)*(om[1]-om[0])
def fixed(p=0.0,E0=0.3):
    w0=omega(E0,p); E1=w0; w1=omega(E1,p)
    for _ in range(5):
        if abs(w1-E1)<2e-4: break
        E2=E1-(w1-E1)*(E1-E0)/((w1-E1)-(w0-E0)); E0,w0,E1=E1,w1,E2; w1=omega(E1,p)
    return E1
E=fixed(); Ed=fixed(1.0,E)
print('непрерывная, тот же метод вставки при энергии E: уровень %.4f, спин %+.5f'%(E,Ed-E))
import sys; sys.exit()
# истинный собственный уровень (без окна): радиальная задача на собственные значения внутри R
import scipy.linalg as sl, scipy.optimize as so
Rb=R; hh=0.02; rr=hh*np.arange(1,int(Rb/hh)+1); Nn=len(rr); UU=-AL/np.sqrt(rr*rr+A*A)
D=(np.diag(np.full(Nn,2.0))-np.diag(np.ones(Nn-1),1)-np.diag(np.ones(Nn-1),-1))/hh**2
lam=lambda o: sl.eigh(D+np.diag(M*M-(o-UU)**2),eigvals_only=True,subset_by_index=[0,0])[0]
print('истинный 1s внутри шара R (Дирихле): %.4f; в бесконечности: '%so.brentq(lam,0.1,M*0.9999),end='')
rr=hh*np.arange(1,int(60/hh)+1); Nn=len(rr); UU=-AL/np.sqrt(rr*rr+A*A); D=(np.diag(np.full(Nn,2.0))-np.diag(np.ones(Nn-1),1)-np.diag(np.ones(Nn-1),-1))/hh**2
print('%.4f'%so.brentq(lambda o: sl.eigh(D+np.diag(M*M-(o-UU)**2),eigvals_only=True,subset_by_index=[0,0])[0],0.1,M*0.9999))
