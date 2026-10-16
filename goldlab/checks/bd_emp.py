"""Ящик из слоёв с весами, подобранными по статистике САМОГО кристалла: A_k(t) — среднее число пар слоя k на расстоянии по времени t;
c_k: Σ c_k A_k(t) ≈ 0 при t > t0 (сокращение вдоль гиперболоидов), c_0 = 1. s — из B(t²) = 2. Голая масса — из средней теории по A(t).
Окно tmax: в сумму идут только y с t_x − t_y ≤ tmax. Сравнение со S1 и непрерывной теорией (шар R, окно H)."""
import sys, math, time, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import atom31f as Af, hinv as Hv
Af.AL=0.6; R=float(sys.argv[1]); H=float(sys.argv[2]); M=float(sys.argv[3]); Af.A=float(sys.argv[4]); NK=int(sys.argv[5]); tmax=float(sys.argv[6]); t0=float(sys.argv[7])
rho=1.0; T0=time.time(); beta=Af.BETA; c0=math.sqrt(rho)/(2*math.pi*math.sqrt(6)); tc2=(2.8*rho**-0.25)**2
X=Af.make_world(rho,H,R); X=X[np.argsort(X[:,0])]; N=len(X); Xf=X.astype(np.float32); t=Xf[:,0]; rr=np.linalg.norm(X[:,1:],axis=1)
def rows(a,b):
    dt=t[None,:]-t[a:b,None]; U=dt*dt
    for k in (1,2,3): U-=(Xf[None,:,k]-Xf[a:b,None,k])**2
    return dt, U
Cf=np.zeros((N,N),np.float32); ch=1500
for a in range(0,N,ch):
    dt,U=rows(a,min(N,a+ch)); Cf[a:a+ch]=((dt>0)&(U>0))
L=np.full((N,N),-1,np.int8); KS=np.zeros((N,N),np.float32)
tb=np.arange(0,tmax+0.25,0.25); Ak=np.zeros((NK,len(tb)-1)); inner=(t>tmax+0.5)&(rr<R-tmax-0.2); ninner=inner.sum()
for a in range(0,N,ch):
    b=min(N,a+ch); dt,U=rows(a,b); C=(dt>0)&(U>0); n=(Cf[a:b]@Cf); lay=np.where(C&(n<NK)&(dt<=tmax),n,-1).astype(np.int8); L[a:b]=lay
    v=(rho*math.pi/24)*U*U; KS[a:b]=np.where((dt>0)&(np.abs(U)<tc2),0.5*c0*np.exp(-v)*np.where(U>0,1-beta,1+beta),0)
    m=(lay>=0)&inner[None,:]; kk=lay[m].astype(int); ti=np.digitize(dt[m],tb)-1; np.add.at(Ak,(kk,ti),1)
del Cf
Ak/=max(ninner,1); tc=0.5*(tb[1:]+tb[:-1])
sel=tc>t0; c=np.r_[1.0,np.linalg.lstsq(Ak[1:,sel].T,-Ak[0,sel],rcond=None)[0]]
A=c@Ak; print('N=',N,'внутренних',ninner,'%.0f с'%(time.time()-T0),' c=',np.round(c,3),flush=True)
print('  A(t) (сумма по слоям, на событие): ',' '.join('%.2f:%+.2f'%(a,b) for a,b in zip(tc[::2],A[::2])),flush=True)
LT=np.ascontiguousarray(L.T); del L; KT=np.ascontiguousarray(KS.T); del KS
cc=np.r_[c,0.0]; W=np.array([cc[LT[x,:x][LT[x,:x]>=0]].sum() for x in range(N)])
def rawB(g):
    out=np.zeros(N)
    for x in np.nonzero(inner)[0]:
        lay=LT[x,:x]; m=lay>=0; out[x]=cc[lay[m]]@(g[:x][m]-g[x])
    return out
s=np.mean(rawB(X[:,0]**2)[inner])/2.0; chk=np.mean(rawB(X[:,1]**2)[inner])/s
Bh=lambda w: (A*(np.exp(1j*w*tc)-1)).sum()/s
mb=-Bh(M).real
print('  s=%.4f, проверка B(x²) = %.3f (должно −2);  W(x) внутр.: %.3f ± %.3f; B̂(ω)/(−ω²) при ω=0.2,0.382,0.7: %s; m_b²=%.4f'%(s,chk,W[inner].mean(),W[inner].std(),' '.join('%.3f%+.3fi'%((Bh(w)/(-w*w)).real,(Bh(w)/(-w*w)).imag) for w in (0.2,0.382,0.7)),mb),flush=True)
at=Af.Atom(X,rho,M,H,R); mbS=at.mb2(M); f=at.src()[:,0].real; Un=-Af.AL/np.sqrt(rr**2+Af.A**2)
def box(Mloc):
    phi=np.zeros(N)
    for x in range(N):
        lay=LT[x,:x]; m=lay>=0; acc=float(cc[lay[m]]@phi[:x][m]) if x else 0.0
        phi[x]=(f[x]-acc/s)/(Mloc[x]-W[x]/s)
    return phi
def s1(Bv):
    phi=np.zeros(N); src=f.copy()
    for j in range(N):
        phi[j]=KT[j,:j]@src[:j]; src[j]+=Bv[j]*phi[j]
    return phi
def modes(phi,lab):
    if not np.all(np.isfinite(phi)) or np.abs(phi).max()>1e10: print('   ',lab,'РАСХОДИТСЯ',flush=True); return None
    Y,_=Af.shell_channels(at,phi.astype(complex)); o,g,w=Hv.pencil(Y,0.5,8)
    print('   ',lab,' '.join('%.3f/%+.3f'%(a,b) for a,b,cw in zip(o,g,w) if a>=0 and cw>0.03*w[0])[:110],flush=True); return o,g,w
modes(box(np.full(N,mb)),'ящик свободно:')
def lev(phi):
    r=modes(phi,'   E-проход')
    if r is None: return np.nan
    o,g,w=r; k=(o>0.05)&(o<M)&(np.abs(g)<0.15); return o[np.nonzero(k)[0][0]] if k.any() else np.nan
E=Af._fixed_point(lambda E: lev(box(mb+2*E*Un-Un*Un)),0.85*M,tol=5e-4,it=5); print('ящик: 1s = %.4f   (S1 на том же мире 0.6527, непрерывная 0.6240 при A=1, M=0.7, R=6)'%E,flush=True)
