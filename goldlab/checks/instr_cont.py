# Тот же измеритель (окно t∈[1.5,19.7], Ханн, проекция на e^{-αr} в шаре r<7) на НЕПРЕРЫВНОЙ КГ во времени.
# Разделяет: что даёт прибор при идеальной физике — и что даёт кристалл.
import numpy as np
al,a=0.45,0.5; h=0.04; R=40.0; r=h*np.arange(1,int(R/h)+1); N=len(r); U=-al/np.sqrt(r*r+a*a)
prof=np.exp(-al*r); ball=r<7
def lap(u,l):
    d=np.empty_like(u); d[1:-1]=(u[2:]-2*u[1:-1]+u[:-2])/h**2; d[0]=(u[1]-2*u[0])/h**2; d[-1]=(-2*u[-1]+u[-2])/h**2
    return d-l*(l+1)/r**2*u
def evolve(m2=1.0,ds=0.0,Ls=4.0,cz=0.0,Lz=2.0,T=19.8,dt=0.01):
    us=np.zeros(N,complex); up=np.zeros(N,complex); vs=(r*prof).astype(complex); vp=np.zeros(N,complex)  # u=rφ, импульс в момент 0
    dS=ds*np.exp(-r/Ls); f=cz*r*np.exp(-r/Lz)/np.sqrt(3)
    def F(y):
        us,up,vs,vp=y
        a_s=2j*U*vs+U*U*us+lap(us,0)-(m2-dS)*us+f*up
        a_p=2j*U*vp+U*U*up+lap(up,1)-(m2-dS)*up+f*us
        return np.array([vs,vp,a_s,a_p])
    y=np.array([us,up,vs,vp]); ts=[]; sig=[]; t=0.0
    while t<T:
        k1=F(y);k2=F(y+dt/2*k1);k3=F(y+dt/2*k2);k4=F(y+dt*k3); y=y+dt/6*(k1+2*k2+2*k3+k4); t+=dt
        ts.append(t); sig.append(np.sum((r*prof*y[0])[ball])*h)   # ∫ r² prof φ dr = ∫ r prof u dr
    return np.array(ts),np.array(sig)
def peak(ts,sig,om=np.linspace(0.6,1.3,1401)):
    s=(ts>1.5)&(ts<19.7); tt=ts[s]; w=np.sin(np.pi*(tt-tt.min())/np.ptp(tt))**2
    S=np.abs(np.exp(-1j*np.outer(om,tt))@(w*sig[s]))**2; i=int(np.argmax(S)); A,B,C=S[i-1],S[i],S[i+1]
    return om[i]+0.5*(A-C)/(A-2*B+C)*(om[1]-om[0])
E0=peak(*evolve()); print('прибор на непрерывной КГ: 1s %.5f (истина 0.90406)'%E0)
for lab,kw,tru in (('m²=0.98',dict(m2=0.98),-0.00932),('m²=0.90',dict(m2=0.90),0.85651-0.90406),('s-вставка L=2',dict(ds=0.02,Ls=2.0),-0.00232),
                   ('s-вставка L=4',dict(ds=0.02,Ls=4.0),-0.00430),('z-вставка L=2',dict(cz=0.12/(2/np.e),Lz=2.0),-0.00399)):
    print('%s: прибор %+.5f, истина %+.5f'%(lab,peak(*evolve(**kw))-E0,tru),flush=True)
