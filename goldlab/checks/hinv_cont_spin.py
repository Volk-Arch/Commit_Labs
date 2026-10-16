import sys, numpy as np; sys.path.insert(0,'/home/claude/goldlab')
from goldlab import hinv as Hv
AL=0.6; A=0.5; M=0.381966; R=11.0; H=20.0
def run(pauli, trunc):
    h=0.04; r=h*np.arange(1,int(44/h)+1); N=len(r); th=(r<R).astype(float) if trunc else np.ones(int(44/h)); U=-AL/np.sqrt(r*r+A*A)*th; prof=np.exp(-AL*M*r)
    m2=M*M*th; fp=pauli*1j*AL*r/(r*r+A*A)**1.5*th
    def lap(u,l):
        d=np.empty_like(u); d[1:-1]=(u[2:]-2*u[1:-1]+u[:-2])/h**2; d[0]=(u[1]-2*u[0])/h**2; d[-1]=(-2*u[-1]+u[-2])/h**2; return d-l*(l+1)/r**2*u
    def F(y):
        us,up,vs,vp=y; return np.array([vs,vp,2j*U*vs+U*U*us+lap(us,0)-m2*us+fp*up,2j*U*vp+U*U*up+lap(up,1)-m2*up+fp*us])
    y=np.array([np.zeros(N,complex),np.zeros(N,complex),(r*prof*th).astype(complex),np.zeros(N,complex)]); t=0.0; dt=0.01; shells=np.linspace(0,R,7); out=[]; ts=[]
    while t<H-0.2:
        k1=F(y);k2=F(y+dt/2*k1);k3=F(y+dt/2*k2);k4=F(y+dt*k3); y=y+dt/6*(k1+2*k2+2*k3+k4); t+=dt
        if abs(t/0.5-round(t/0.5))<1e-6: ts.append(t); out.append([np.sum((r*y[0])[(r>=a)&(r<b)])*h for a,b in zip(shells[:-1],shells[1:])])
    ts=np.array(ts); Y=np.array(out).T; s=ts>1.5; return Hv.level(Y[:,s],0.5,0.05,M)[0]
for tr in (False,True):
    E=run(0,tr); Ed=run(1,tr); print('%s: уровень %.4f, спин %+.5f'%('источник и поле внутри R' if tr else 'без обрезки',E,Ed-E),flush=True)
import scipy.linalg as sl, scipy.optimize as so
