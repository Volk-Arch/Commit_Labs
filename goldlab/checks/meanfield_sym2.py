import numpy as np, math
from meanfield31 import root
def delta_shape(rho, hfun, T=10.0, N=2000):
    c=math.sqrt(rho)/(2*math.pi*math.sqrt(6)); al=rho*math.pi/24; t=np.linspace(1e-6,T,N); A=np.empty(N)
    for i,tt in enumerate(t):
        r=np.linspace(0,math.sqrt(tt*tt+7/math.sqrt(al)),6001); u=tt*tt-r*r; A[i]=np.trapezoid(4*np.pi*r*r*hfun(al*u*u),r)
    return t, 0.5*c*A-t
for nm,h in (('гаусс',lambda x: np.exp(-x)),('гаусс без δ″: (3/2 − αu²)e^{−αu²}',lambda x:(1.5-x)*np.exp(-x))):
    for rho in (1.0,4.0):
        t,d=delta_shape(rho,h); s='%s ρ=%g: '%(nm,rho)
        for m in (0.5,1.0,1.41):
            w=root(t,d,m); s+='m=%.2f: ω=%.3f%+.4fi; '%(m,w.real,w.imag)
        print(s,flush=True)
