"""Наклон размытия: вес (1 − β) внутри конуса и (1 + β) снаружи (несравнимые). β > 0 даёт отрицательный первый момент — затухание,
которое может погасить раскачку от второго момента. Ищем β, при котором Im ω = 0 у m = 1."""
import numpy as np, math, scipy.optimize as so
from meanfield31 import root
def delta_tilt(rho,beta,T=10.0,N=1500):
    c=math.sqrt(rho)/(2*math.pi*math.sqrt(6)); al=rho*math.pi/24; t=np.linspace(1e-6,T,N); A=np.empty(N)
    for i,tt in enumerate(t):
        r=np.linspace(0,math.sqrt(tt*tt+6/math.sqrt(al)),5001); u=tt*tt-r*r; A[i]=np.trapezoid(4*np.pi*r*r*np.exp(-al*u*u)*np.where(u>0,1-beta,1+beta),r)
    return t,0.5*c*A-t
for rho in (1.0,2.0,4.0):
    f=lambda b: root(*delta_tilt(rho,b),1.0).imag
    b=so.brentq(f,0.0,0.3,xtol=1e-4); t,d=delta_tilt(rho,b)
    print('ρ=%g: β = %.4f; m=0.5: %s; m=1: %s; m=1.41: %s'%(rho,b,*[np.round(root(t,d,m),4) for m in (0.5,1.0,1.41)]),flush=True)
