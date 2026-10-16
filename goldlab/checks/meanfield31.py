"""Почему разгон: средняя (по посыпке) связь — ядро c·e^{−ρV(τ)}, V = πτ⁴/24. При k = 0 её образ F(ω) = −1/ω² + G(ω),
G — от недостачи ядра на малых временах (δ(t) = c·A(t) − t). Мода массы: 1 + m²F(ω) = 0. Корень с Im ω > 0 — рост e^{(Im ω)t}."""
import numpy as np, math, scipy.optimize as so
def delta(h, rho=1.0, T=8.0, N=4000):
    c=math.sqrt(rho)/(2*math.pi*math.sqrt(6)); t=np.linspace(1e-6,T,N); A=np.empty(N)
    for i,tt in enumerate(t):
        r=np.linspace(0,tt,4001); A[i]=np.trapezoid(4*np.pi*r*r*h(rho*np.pi*(tt*tt-r*r)**2/24),r)
    return t, c*A-t
def root(t,d,m,w0=None):
    G=lambda w: np.trapezoid(np.exp(1j*w*t)*d,t)
    f=lambda v: (lambda w: [ (1+m*m*(-1/w**2+G(w))).real, (1+m*m*(-1/w**2+G(w))).imag ])(v[0]+1j*v[1])
    s=so.fsolve(f,[m if w0 is None else w0,0.0],full_output=True); return s[0][0]+1j*s[0][1]
if __name__=='__main__':
    links=lambda s: np.exp(-s)
    for rho in (1.0,2.0,4.0,8.0):
        t,d=delta(links,rho); print('ρ=%g: ∫δ dt = %.4f, ∫tδ dt = %.4f'%(rho,np.trapezoid(d,t),np.trapezoid(t*d,t)))
        for m in (0.5,0.71,1.0,1.41):
            w=root(t,d,m); print('   m=%.2f: ω = %.4f %+.4fi  (рост %+.3f)'%(m,w.real,w.imag,w.imag))
