"""Ящик □ из слоёв прошлого на кристалле (плотная проверка, шар R, окно H). Слой k: y ≺ x и в интервале ровно k событий.
B φ(x) = (1/s)[−α φ(x) + Σ_k c_k Σ_{слой k} φ(y)]; c_k — из обнуления моментов M0..M3 (5 слоёв); марш: φ(x) = (f − (1/s)Σ…)/(m_b² − α/s) ...
Сравнение: Пуассоновы веса (как у Соркина) и веса из распределения слоёв самого кристалла."""
import sys, math, time, json, numpy as np
sys.path.insert(0,'/home/claude/goldlab'); sys.path.insert(0,'/home/claude/code/goldlab_checks')
from goldlab import atom31f as Af, hinv as Hv
import bd_mean as BM
Af.AL=0.6; R=float(sys.argv[1]); H=float(sys.argv[2]); Ms=[float(x) for x in sys.argv[3].split(',')]; NK=int(sys.argv[4]) if len(sys.argv)>4 else 5; kind=sys.argv[5] if len(sys.argv)>5 else 'gold'
rho=1.0; T0=time.time()
X=Af.make_world(rho,H,R) if kind=='gold' else Af.make_world(rho,H,R,poisson_seed=int(kind[1:] or 1)); X=X[np.argsort(X[:,0])]; N=len(X)
Xf=X.astype(np.float32); t=Xf[:,0]
def rows(a,b):
    dt=t[None,:]-t[a:b,None]; U=dt*dt
    for k in (1,2,3): U-=(Xf[None,:,k]-Xf[a:b,None,k])**2
    return dt>0, U
Cf=np.zeros((N,N),np.float32); ch=1500
for a in range(0,N,ch):
    fu,U=rows(a,min(N,a+ch)); Cf[a:a+ch]=(fu&(U>0))
L=np.full((N,N),-1,np.int8)        # L[y,x] = номер слоя (0..NK−1) или −1
vb=np.linspace(0,16,321); H_=np.zeros((NK+1,len(vb)-1))
for a in range(0,N,ch):
    b=min(N,a+ch); fu,U=rows(a,b); C=fu&(U>0); n=(Cf[a:b]@Cf); v=(rho*math.pi/24)*U*U
    lay=np.where(C&(n<NK),n,-1).astype(np.int8); L[a:b]=lay
    m=C&(v<16); kk=np.minimum(n[m],NK).astype(int); vi=np.digitize(v[m],vb)-1
    np.add.at(H_,(kk,vi),1)
del Cf
Pt=H_/np.maximum(H_.sum(0),1)            # условная доля
Pd=H_/np.maximum(H_.sum(0).reshape(1,-1).sum()/ (len(vb)-1),1)*0+H_/np.maximum(np.convolve(H_.sum(0),np.ones(21)/21,'same'),1)  # плотность: сглаженная по 1 ед. v
vc=0.5*(vb[1:]+vb[:-1])
Pcr=[(lambda k: (lambda v: np.interp(v,vc,np.convolve(Pd[k],np.ones(21)/21,'same'),right=0.0)))(k) for k in range(NK)]
Ppo=[BM.Pk_poisson(k) for k in range(NK)]
print(kind,'N=',N,'%.0f с'%(time.time()-T0),' n̄ по v=0.5,1,2,3,4:',[round(float((np.arange(NK+1)[:,None]*Pt).sum(0)[np.argmin(abs(vc-v))]),2) for v in (0.5,1,2,3,4)],flush=True)
boxes={}
for nm,P in (('Пуассон',Ppo),('кристалл',Pcr)):
    c=BM.design(P,NK); F0,Fs=BM.Bhat(c,P,np.r_[0.05,Ms],T=80,N=6000); B=Fs-F0; s=-(B[0].real)/0.05**2; Bn=B[1:]/s
    boxes[nm]=dict(c=c,alpha=F0,s=s,mb2={M:-float(b.real) for M,b in zip(Ms,Bn)},im={M:float(b.imag) for M,b in zip(Ms,Bn)})
    print(nm,'c=',np.round(c,2),'α=%.4f s=%.4f'%(F0,s),' m_b²:',{M:round(v,4) for M,v in boxes[nm]['mb2'].items()},' ImB̂:',{M:round(v,4) for M,v in boxes[nm]['im'].items()},flush=True)
Un=-Af.AL/np.sqrt(np.linalg.norm(X[:,1:],axis=1)**2+Af.A**2)
LT=np.ascontiguousarray(L.T); del L     # LT[x,y]
LOCAL=len(sys.argv)>6 and sys.argv[6]=='local'
def march(bx,Mloc,f):
    c=np.r_[bx['c'],0.0]; phi=np.zeros(N); s=bx['s']; al_=bx['alpha']
    for x in range(N):
        lay=LT[x,:x]; m=lay>=0; w=c[lay[m]]; acc=float(w@phi[:x][m]) if x else 0.0
        a_=float(w.sum()) if LOCAL else al_
        phi[x]=(f[x]-acc/s)/(Mloc[x]-a_/s)
    return phi
res={}
for M in Ms:
    at=Af.Atom(X,rho,M,H,R); f=at.src()[:,0].real
    for nm,bx in boxes.items():
        mb=bx['mb2'][M]
        phi=march(bx,np.full(N,mb),f)
        if not np.all(np.isfinite(phi)) or np.abs(phi).max()>1e8: print('M=%.3f %s свободно: РАСХОДИТСЯ'%(M,nm),flush=True); continue
        Y,_=Af.shell_channels(at,phi.astype(complex)); o,g,w=Hv.pencil(Y,0.5,8)
        print('M=%.3f %-9s свободно моды:'%(M,nm),' '.join('%.3f/%+.3f'%(a,b) for a,b,cw in zip(o,g,w) if a>0 and cw>0.03*w[0])[:100],flush=True)
        def om(E):
            ph=march(bx,mb+2*E*Un-Un*Un,f)
            if not np.all(np.isfinite(ph)): return np.nan
            Y,_=Af.shell_channels(at,ph.astype(complex)); o,g,w=Hv.pencil(Y,0.5,8); k=(o>0.05)&(o<M)&(np.abs(g)<0.15)
            print('     E=%.4f моды:'%E,' '.join('%.3f/%+.3f'%(a,b) for a,b,cw in zip(o,g,w) if a>0 and cw>0.03*w[0])[:90],flush=True)
            return o[np.nonzero(k)[0][0]] if k.any() else np.nan
        E=Af._fixed_point(om,0.85*M,tol=5e-4,it=5); res[(M,nm)]=E; print('M=%.3f %-9s 1s = %.4f'%(M,nm,E),flush=True)
    ref=Af.cont_level_hinv(M,R,H); print('M=%.3f непрерывная (обрезка) %.4f'%(M,ref),flush=True)
