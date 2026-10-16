"""Шаг 2в: односторонний конус (только сравнимые пары, вес по счёту интервала) с теми же моментами M0, M1, что у S1. Средняя теория."""
import math, numpy as np
rho=1.0; beta=0.115; c=math.sqrt(rho)/(2*math.pi*math.sqrt(6)); al=rho*math.pi/24; sa=math.sqrt(al)
lam=math.sqrt(math.pi)*(1+beta)/(math.pi/2+beta); a=2*math.sqrt(math.pi)/(math.sqrt(math.pi)-lam)
def hS1(u): return 0.5*c*np.exp(-al*u*u)*np.where(u>0,1-beta,1+beta)
def hOS(u): return np.where(u>0,0.5*c*a*np.exp(-al*u*u)*(1-lam*sa*u),0.0)
def delta(h,T=16.0,N=800):
    t=np.linspace(1e-6,T,N); A=np.empty(N)
    for i,tt in enumerate(t):
        uu=np.linspace(-9/sa,tt*tt,200001); A[i]=np.trapezoid(2*np.pi*np.sqrt(np.maximum(tt*tt-uu,0))*h(uu),uu)
    return t,A-t
print('λ′=%.4f a=%.4f, смена знака при ρV=%.3f'%(lam,a,1/lam**2))
for nm,h in (('S1',hS1),('односторонний',hOS)):
    t,d=delta(h); om=np.array([0.3,0.5,0.7,1.0,1.3]); G=[np.trapezoid(np.exp(1j*w*t)*d,t) for w in om]
    print(nm,' t·δ(t) при t=4,8,12:',' '.join('%.4f'%(tt*d[np.argmin(abs(t-tt))]) for tt in (4,8,12)))
    print('   G(ω):',' '.join('%.3f%+.3fi'%(g.real,g.imag) for g in G))
