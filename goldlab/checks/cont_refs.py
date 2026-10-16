import sys; H=float(sys.argv[1]); Rb=float(sys.argv[2])
src=open('instr_cont.py').read().split('E0=peak')[0].replace('ball=r<7','ball=r<%g'%Rb).replace('T=19.8','T=%g'%(H-0.2)).replace('ts<19.7','ts<%g'%(H-0.3))
exec(src)
E1=peak(*evolve()); E2=peak(*evolve(m2=0.81)); Es=peak(*evolve(cz=0.12/(2/np.e),Lz=2.0))
print('непрерывная КГ, прибор H=%g R=%g: 1s %.5f (истина 0.90406), наклон %.3f (истина 0.487), штарк %+.5f (истина −0.00399)'%(H,Rb,E1,(E1-E2)/0.19,Es-E1))
# Дирак тем же прибором: s½ ↔ p½ через iσ·∇U (σ·r̂ переводит s½ в p½ с единичным элементом)
import re
dU=al*r/(r*r+a*a)**1.5
def evolve_d(T=H-0.2,dt=0.01,cc=1j):
    us=np.zeros(N,complex); up=np.zeros(N,complex); vs=(r*prof).astype(complex); vp=np.zeros(N,complex); f=cc*dU
    def F(y):
        us,up,vs,vp=y
        return np.array([vs,vp,2j*U*vs+U*U*us+lap(us,0)-us+f*up,2j*U*vp+U*U*up+lap(up,1)-up+f*us])
    y=np.array([us,up,vs,vp]); ts=[]; sig=[]; t=0.0
    while t<T:
        k1=F(y);k2=F(y+dt/2*k1);k3=F(y+dt/2*k2);k4=F(y+dt*k3); y=y+dt/6*(k1+2*k2+2*k3+k4); t+=dt; ts.append(t); sig.append(np.sum((r*prof*y[0])[ball])*h)
    return np.array(ts),np.array(sig)
print('   Дирак тем же прибором: сдвиг %+.5f (истина +0.0065)'%(peak(*evolve_d())-E1))
