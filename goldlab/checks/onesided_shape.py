import math, numpy as np
from math import gamma, sqrt, pi
from onesided_scan import G, c, al, sa, hS1
I=lambda k: gamma((k+1)/2)/2
def mk(p,bp):
    # h = (c/2) e^{-s²} s^p (a - b s), s = √α u; M0: a I_p - b I_{p+1} = √π ; M1: a I_{p+1} - b I_{p+2} = -β′
    A=np.array([[I(p),-I(p+1)],[I(p+1),-I(p+2)]]); a,b=np.linalg.solve(A,[sqrt(pi),-bp])
    return (a,b),(lambda u: np.where(u>0,0.5*c*np.exp(-al*u*u)*(sa*np.maximum(u,0))**p*(a-b*sa*np.maximum(u,0)),0.0))
oms=[0.382,0.7,1.0]
g,_,_=G(hS1,oms); print('S1:',' '.join('%.3f%+.3fi'%(x.real,x.imag) for x in g),flush=True)
for p in (1,2):
    for bp in (-0.1,0.0,0.1,0.2):
        ab,h=mk(p,bp); g,_,_=G(h,oms); print('p=%d β′=%+.2f a,b=(%.2f,%.2f):'%(p,bp,*ab),' '.join('%.3f%+.3fi'%(x.real,x.imag) for x in g),flush=True)
