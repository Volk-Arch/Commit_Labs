"""Ящик □ из слоёв прошлого (Бенинкаса–Даукер / Соркин), средняя теория. B φ(x) = −α φ(x) + Σ_{y≺x} b(n_xy) φ(y), n — число событий в интервале.
Среднее ядро: b(v) = Σ_k c_k P_k(v), v = ρV, P_k(v) — вероятность того, что в интервале объёма v ровно k событий (Пуассон или кристалл).
F(ω) = ∫ e^{iωt} A(t) dt, A(t) = ∫ d³x b; Ĉ(ω) = −α + F(ω) должно быть −ω² (□e^{−iωt} = −ω²): Im B̂(ω) > 0 при ω = m — раскачка."""
import sys, math, numpy as np
rho=1.0; al=rho*math.pi/24
UMAX=(16/al)**0.5; UG=np.linspace(0,UMAX,40001); WG=np.full(len(UG),UG[1]); WG[0]=WG[-1]=UG[1]/2
def A_of(bfun, t):
    b=bfun(al*UG*UG); out=np.empty(len(t))
    for i,tt in enumerate(t):
        if tt*tt>UMAX*1.0001:   # вычитаем асимптотику: моменты M0..M2 = 0 точно на этой сетке
            r=np.sqrt(tt*tt-UG)-tt+UG/(2*tt)+UG**2/(8*tt**3); out[i]=2*np.pi*np.sum(WG*b*r)
        else:
            out[i]=2*np.pi*np.sum(WG*b*np.sqrt(np.maximum(tt*tt-UG,0)))
    return out
def moments(bfun,K=4):
    b=bfun(al*UG*UG); return np.array([np.sum(WG*b*UG**k) for k in range(K)])
def Pk_poisson(k): return lambda v: v**k*np.exp(-v)/math.factorial(k)
def design(P, nk, extra=None):
    """c_k: M0 = 0, M1 = 0 (+ M2 = 0 если nk = 4), нормировка — потом по ω²."""
    Ms=np.array([moments(P[k]) for k in range(nk)]).T   # Ms[j,k] = M_j(P_k)
    rows=nk-1; A=Ms[:rows,:nk]; ns=np.linalg.svd(A)[2][-1]; c=ns/ns[0]; return c
def Bhat(c,P,oms,T=60.0,N=3000):
    t=np.linspace(1e-4,T,N); bf=lambda v: sum(ck*P[k](v) for k,ck in enumerate(c)); A=A_of(bf,t)
    F=lambda w: np.trapezoid(np.exp(1j*w*t)*A*np.exp(-(t/T*3)**2),t)
    F0=F(0.0).real; Fs=np.array([F(w) for w in oms]); return F0, Fs
if __name__=='__main__':
    P=[Pk_poisson(k) for k in range(4)]
    bd=np.array([1,-9,16,-8.]); print('БД (1,−9,16,−8): моменты M0..M3 =',moments(lambda v: sum(ck*P[k](v) for k,ck in enumerate(bd))).round(4))
    for nk in (3,4):
        c=design(P,nk); print('Пуассон, %d слоя: c ='%nk, (c/c[0]).round(3))
    oms=np.array([0.1,0.2,0.382,0.7,1.0,1.4])
    for nm,c in (('БД',bd),):
        F0,Fs=Bhat(c,P,oms); al_=F0; B=Fs-F0; norm=-(B[0].real)/oms[0]**2
        print(nm,'α=%.4f норма=%.4f'%(al_,norm)); 
        for w,b in zip(oms,B/norm): print('   ω=%.3f  B̂/норма = %+.4f %+.4fi   (−ω² = %+.4f)'%(w,b.real,b.imag,-w*w))
