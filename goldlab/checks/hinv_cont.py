"""Проверка прибора на непрерывной теории: многоканальная гармоническая инверсия в коротком окне против истинного уровня."""
import sys, numpy as np; sys.path.insert(0,'/home/claude/goldlab')
from goldlab import hinv as Hv
AL=0.6; A=0.5; M=0.381966; R=11.0
def run(H, src_inside=False, trunc=False, m=M):
    h=0.04; r=h*np.arange(1,int(44/h)+1); N=len(r); th=(r<R).astype(float); U=-AL/np.sqrt(r*r+A*A)*(th if trunc else 1); prof=np.exp(-AL*m*r)
    m2=m*m*(th if trunc else 1)
    def lap(u):
        d=np.empty_like(u); d[1:-1]=(u[2:]-2*u[1:-1]+u[:-2])/h**2; d[0]=(u[1]-2*u[0])/h**2; d[-1]=(-2*u[-1]+u[-2])/h**2; return d
    def F(y):
        us,vs=y; return np.array([vs,2j*U*vs+U*U*us+lap(us)-m2*us])
    y=np.array([np.zeros(N,complex),(r*prof*(th if src_inside else 1)).astype(complex)]); t=0.0; dt=0.01; shells=np.linspace(0,R,7); out=[]; ts=[]
    while t<H-0.2:
        k1=F(y);k2=F(y+dt/2*k1);k3=F(y+dt/2*k2);k4=F(y+dt*k3); y=y+dt/6*(k1+2*k2+2*k3+k4); t+=dt
        if abs(t/0.25-round(t/0.25))<1e-6:
            ts.append(t); out.append([np.sum((r*y[0])[(r>=a)&(r<b)])*h for a,b in zip(shells[:-1],shells[1:])])
    return np.array(ts), np.array(out).T
import scipy.linalg as sl, scipy.optimize as so
hh=0.02; rr=hh*np.arange(1,int(60/hh)+1); UU=-AL/np.sqrt(rr*rr+A*A); D=(np.diag(np.full(len(rr),2.0))-np.diag(np.ones(len(rr)-1),1)-np.diag(np.ones(len(rr)-1),-1))/hh**2
true=so.brentq(lambda o: sl.eigh(D+np.diag(M*M-(o-UU)**2),eigvals_only=True,subset_by_index=[0,0])[0],0.1,M*0.9999)
print('истинный уровень 1s: %.4f (порог m = %.4f)'%(true,M))
for H in (12,20,30):
    for si,tr in ((False,False),(True,True)):
        ts,Y=run(H,si,tr); s=ts>1.5; E,(om,ga,w)=Hv.level(Y[:,s],0.25,0.05,M)
        print('окно %2d, %s: инверсия → %.4f; моды (ω, γ): %s'%(H,'источник и поле внутри R' if si else 'без обрезки',E,' '.join('(%.3f,%.3f)'%(a,b) for a,b in zip(om[:5],ga[:5]))),flush=True)
