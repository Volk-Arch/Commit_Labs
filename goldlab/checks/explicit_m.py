"""Массивный пропагатор 3+1 БЕЗ цепочки по массе: G_m = δ(τ²)/2π − θ(τ²)·m J1(mτ)/(4πτ). Световая часть — связи (одно применение,
сравнимые пары с пустым интервалом), внутренняя — по всем сравнимым парам, τ — из счёта интервала (τ⁴ = 24 v/π, таблица n → v).
Потенциал — цепочкой вставок V = 2EU − U² при энергии E. Только сравнимые пары; несравнимые не нужны."""
import sys, math, time, numpy as np
from scipy.special import j1
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import atom31f as Af, hinv as Hv
Af.AL=0.6; R=float(sys.argv[1]); H=float(sys.argv[2]); M=float(sys.argv[3]); Af.A=float(sys.argv[4]); mode=sys.argv[5] if len(sys.argv)>5 else 'order'
rho=1.0; T0=time.time(); coef=math.sqrt(rho)/(2*math.pi*math.sqrt(6)); al=rho*math.pi/24
X=Af.make_world(rho,H,R); X=X[np.argsort(X[:,0])]; N=len(X); Xf=X.astype(np.float32); t=Xf[:,0]; rr=np.linalg.norm(X[:,1:],axis=1)
def rows(a,b):
    dt=t[None,:]-t[a:b,None]; U=dt*dt
    for k in (1,2,3): U-=(Xf[None,:,k]-Xf[a:b,None,k])**2
    return dt, U
Cf=np.zeros((N,N),np.float32); ch=1500
for a in range(0,N,ch):
    dt,U=rows(a,min(N,a+ch)); Cf[a:a+ch]=((dt>0)&(U>0))
# таблица n → v (калибровка кристалла)
sv=np.zeros(4001); cv=np.zeros(4001)
for a in range(0,N,ch):
    b=min(N,a+ch); dt,U=rows(a,b); C=(dt>0)&(U>0); n=np.minimum(Cf[a:b]@Cf,4000).astype(int); v=al*U*U
    sv+=np.bincount(n[C],v[C],4001); cv+=np.bincount(n[C],minlength=4001)
vtab=np.where(cv>20,sv/np.maximum(cv,1),np.nan); kk=np.nonzero(np.isfinite(vtab))[0]; vtab=np.interp(np.arange(4001),kk,vtab[kk])
K=np.zeros((N,N),np.float32)
for a in range(0,N,ch):
    b=min(N,a+ch); dt,U=rows(a,b); C=(dt>0)&(U>0); n=np.minimum(Cf[a:b]@Cf,4000).astype(int)
    tau=np.sqrt(U.clip(0)) if mode=='coord' else (24*vtab[n]/(math.pi*rho))**0.25
    tau=np.maximum(tau,1e-6); inner=-M*j1(M*tau)/(4*np.pi*tau)
    if mode=='s1':
        v=al*U*U; K[a:b]=np.where((dt>0)&(np.abs(U)<(2.8*rho**-0.25)**2),0.5*coef*np.exp(-v)*np.where(U>0,1-Af.BETA,1+Af.BETA),0)
    else: K[a:b]=np.where(C&(n==0),coef,0)+np.where(C,inner,0)
del Cf
KT=np.ascontiguousarray(K.T); del K
at=Af.Atom(X,rho,M,H,R); f=at.src()[:,0].real; Un=-Af.AL/np.sqrt(rr**2+Af.A**2); MB=at.mb2(M) if mode=='s1' else 0.0
print(mode,'N=',N,'%.0f с'%(time.time()-T0),flush=True)
def run(Bv):
    phi=np.zeros(N); src=f.copy()
    for j in range(N):
        phi[j]=KT[j,:j]@src[:j]; src[j]+=Bv[j]*phi[j]
    return phi
def lev(phi,lab,hi):
    if not np.all(np.isfinite(phi)) or np.abs(phi).max()>1e12: print('   ',lab,'РАСХОДИТСЯ',flush=True); return np.nan
    Y,_=Af.shell_channels(at,phi.astype(complex)); o,g,w=Hv.pencil(Y,0.5,8); k=(o>0.05)&(o<hi)&(np.abs(g)<0.15)
    print('   ',lab,' '.join('%.3f/%+.3f'%(a,b) for a,b,cw in zip(o,g,w) if a>=0 and cw>0.03*w[0])[:100],flush=True)
    return o[np.nonzero(k)[0][0]] if k.any() else np.nan
def runs(b,pauli):
    Bm=(b[:,None,None]*at.eye+pauli*1j*at.dU[:,None,None]*at.sr)/rho; phi=np.zeros((N,2),complex); src=np.array(at.src(),complex)
    for j in range(N):
        phi[j]=KT[j,:j]@src[:j]; src[j]+=Bm[j]@phi[j]
    return phi[:,0]
lev(run(np.full(N,-MB/rho)),'свободно (одно применение):',3.0)
E=Af._fixed_point(lambda E: lev(run(-(MB+2*E*Un-Un*Un)/rho),'E-проход',M),0.85*M,tol=5e-4,it=6); print('%s: 1s = %.4f   (S1 0.6527, непрерывная 0.6240; свободная в шаре — непрерывная 0.857)'%(mode,E),flush=True)
if len(sys.argv)>6:
    Es=Af._fixed_point(lambda x: lev(runs(-(MB+2*x*Un-Un*Un),1.0),'спин',M),E,tol=3e-4,it=6)-E
    ref=Af.cont_level_hinv(M,R,H); refs=Af.cont_level_hinv(M,R,H,pauli=1.0)-ref
    print('%s: спиновый сдвиг %+.5f (непрерывная %+.5f; 1s непрерывная %.4f)'%(mode,Es,refs,ref),flush=True)
