"""Симметрично размытый конус: ядро c′·e^{−α u²} по обе стороны (u = τ², сравнимые и несравнимые пары, t_x > t_y).
Первый момент по u нулевой ⇒ нет δ′. Средняя теория: хвост и корень моды массы."""
import numpy as np, math
from meanfield31 import root
def delta_sym(rho, T=10.0, N=2500, R=None, shape='gauss'):
    c=math.sqrt(rho)/(2*math.pi*math.sqrt(6)); al=rho*math.pi/24; t=np.linspace(1e-6,T,N); A=np.empty(N)
    for i,tt in enumerate(t):
        rmax=math.sqrt(tt*tt+6/math.sqrt(al))                 # до |u| ≈ 6/√α (за пределом вес ничтожен)
        r=np.linspace(0,rmax,6001); u=tt*tt-r*r; A[i]=np.trapezoid(4*np.pi*r*r*np.exp(-al*u*u),r)
    return t, 0.5*c*A-t                                       # c′ = c/2: обе стороны вместе дают δ(τ²)/(2π)
for rho in (1.0,2.0,4.0):
    t,d=delta_sym(rho); s='ρ=%g: хвост t·δ %+.4f; ∫δ dt %+.3f; '%(rho,(t*d)[-200:].mean(),np.trapezoid(d,t))
    for m in (0.5,1.0,1.41,2.0):
        w=root(t,d,m); s+='m=%.2f: ω=%.3f%+.4fi; '%(m,w.real,w.imag)
    print(s,flush=True)
