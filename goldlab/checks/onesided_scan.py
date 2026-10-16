import math, numpy as np
rho=1.0; c=math.sqrt(rho)/(2*math.pi*math.sqrt(6)); al=rho*math.pi/24; sa=math.sqrt(al)
def mk(bp):
    lam=math.sqrt(math.pi)*(1+bp)/(math.pi/2+bp); a=2*math.sqrt(math.pi)/(math.sqrt(math.pi)-lam)
    return lambda u: np.where(u>0,0.5*c*a*np.exp(-al*u*u)*(1-lam*sa*u),0.0)
def hS1(u,beta=0.115): return 0.5*c*np.exp(-al*u*u)*np.where(u>0,1-beta,1+beta)
def G(h,oms,T=40.0,N=1600):
    t=np.linspace(1e-6,T,N); A=np.empty(N)
    for i,tt in enumerate(t):
        uu=np.linspace(-9/sa,min(tt*tt,60.0),60001); A[i]=np.trapezoid(2*np.pi*np.sqrt(np.maximum(tt*tt-uu,0))*h(uu),uu)
    d=A-t; win=np.exp(-(t/T*2.5)**2)    # мягкое окно против логарифмического хвоста
    return [np.trapezoid(np.exp(1j*w*t)*d*win,t) for w in oms], t, d
if __name__=='__main__':
    oms=[0.382,0.7,1.0]
    g,_,_=G(hS1,oms); print('S1 β=0.115:',' '.join('%.3f%+.3fi'%(x.real,x.imag) for x in g))
    g,_,_=G(lambda u:hS1(u,0.0),oms); print('S1 β=0:',' '.join('%.3f%+.3fi'%(x.real,x.imag) for x in g))
    for bp in (-0.3,-0.15,0.0,0.115,0.3,0.5):
        g,_,_=G(mk(bp),oms); print('односторонний β′=%+.3f:'%bp,' '.join('%.3f%+.3fi'%(x.real,x.imag) for x in g))
    from math import gamma, sqrt, pi
    I=lambda k: gamma((k+1)/2)/2
    def mk3(bp):
        Mx=np.array([[I(k+n) for k in range(3)] for n in range(3)]); a=np.linalg.solve(Mx,[sqrt(pi),-bp,sqrt(pi)/2])
        return a, (lambda u: np.where(u>0,0.5*c*np.exp(-al*u*u)*(a[0]+a[1]*sa*u+a[2]*al*u*u),0.0))
    print('--- три момента (M0, M1(β′), M2 как у S1)')
    from math import gamma, sqrt, pi
    I=lambda k: gamma((k+1)/2)/2
    def mk3(bp):
        Mx=np.array([[I(k+n) for k in range(3)] for n in range(3)]); a=np.linalg.solve(Mx,[sqrt(pi),-bp,sqrt(pi)/2])
        return a, (lambda u: np.where(u>0,0.5*c*np.exp(-al*u*u)*(a[0]+a[1]*sa*u+a[2]*al*u*u),0.0))
    for bp in (0.0,0.05,0.115,0.2):
        a,h=mk3(bp); g,_,_=G(h,oms); print('β′=%+.3f a=(%.2f,%.2f,%.2f):'%(bp,*a),' '.join('%.3f%+.3fi'%(x.real,x.imag) for x in g))
