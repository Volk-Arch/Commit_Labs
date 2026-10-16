"""Симметричный конус БЕЗ координат в весах: сравнимые пары — e^{−v(n)}, n — счёт интервала; несравнимые — e^{−v(n_min)}, n_min — наименьший
ромб с вершинами-событиями; порядок обхода — высота (длина самой длинной цепи снизу), а не время. Координаты — только для поиска кандидатов
(щедрое окно |τ²| < ucut) и для разовой калибровки таблиц кристалла. Сравнение со S1 по координатам и с непрерывной теорией."""
import sys, math, time, numpy as np, numba
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import atom31f as Af, hinv as Hv
Af.AL=0.6; R=float(sys.argv[1]); H=float(sys.argv[2]); M=float(sys.argv[3]); Af.A=float(sys.argv[4]); CAP=12
rho=1.0; T0=time.time(); beta=Af.BETA; c0=math.sqrt(rho)/(2*math.pi*math.sqrt(6)); tc2=(2.8*rho**-0.25)**2; ucut=12.0; al=rho*math.pi/24
X=Af.make_world(rho,H,R); X=X[np.argsort(X[:,0])]; N=len(X); Xf=X.astype(np.float32); t=Xf[:,0]; rr=np.linalg.norm(X[:,1:],axis=1)
def rows(a,b):
    dt=t[None,:]-t[a:b,None]; U=dt*dt
    for k in (1,2,3): U-=(Xf[None,:,k]-Xf[a:b,None,k])**2
    return dt, U
Cf=np.zeros((N,N),np.float32); ch=1500
for a in range(0,N,ch):
    dt,U=rows(a,min(N,a+ch)); Cf[a:a+ch]=((dt>0)&(U>0))
Cnt=np.full((N,N),-1,np.int16)
for a in range(0,N,ch):
    b=min(N,a+ch); n=Cf[a:b]@Cf; Cnt[a:b]=np.where(Cf[a:b]>0,np.minimum(n,32000),-1).astype(np.int16)
# высота: самая длинная цепь снизу
h=np.zeros(N,np.int32)
CfT=np.ascontiguousarray(Cf.T)
for x in range(N):
    p=np.nonzero(CfT[x,:x])[0]; h[x]=h[p].max()+1 if len(p) else 0
del CfT
del Cf
print('N=',N,'матрицы %.0f с, высота 0..%d'%(time.time()-T0,h.max()),flush=True)
@numba.njit
def csr_near(Cnt,cap,past):
    n=Cnt.shape[0]; ptr=np.zeros(n+1,np.int64)
    for i in range(n):
        c=0
        for p in range(n):
            v=Cnt[p,i] if past else Cnt[i,p]
            if v>=0 and v<=cap: c+=1
        ptr[i+1]=ptr[i]+c
    idx=np.empty(ptr[n],np.int32)
    for i in range(n):
        w=ptr[i]
        for p in range(n):
            v=Cnt[p,i] if past else Cnt[i,p]
            if v>=0 and v<=cap: idx[w]=p; w+=1
    return ptr,idx
NPp,NPi=csr_near(Cnt,CAP,True); NFp,NFi=csr_near(Cnt,CAP,False)
print('ближнее прошлое: в среднем %.0f'%((NPp[1:]-NPp[:-1]).mean()),'%.0f с'%(time.time()-T0),flush=True)
@numba.njit
def spacelike(X,Cnt,NPp,NPi,NFp,NFi,ucut,cap):
    n=X.shape[0]; I=[]; J=[]; NM=[]; UU=[]; Qb=np.empty(n,np.int64)
    for i in range(n):
        for j in range(i+1,n):
            if Cnt[i,j]>=0 or Cnt[j,i]>=0: continue
            dt=X[j,0]-X[i,0]; u=dt*dt-(X[j,1]-X[i,1])**2-(X[j,2]-X[i,2])**2-(X[j,3]-X[i,3])**2
            if u<=-ucut: continue
            best=1<<30; nq=0
            for b in range(NFp[i],NFp[i+1]):
                q=NFi[b]; vq=Cnt[j,q]
                if vq>=0 and vq<=cap: Qb[nq]=q; nq+=1
            if nq==0: continue
            for a in range(NPp[i],NPp[i+1]):
                p=NPi[a]; vp=Cnt[p,j]
                if vp<0 or vp>cap: continue
                for b in range(nq):
                    c=Cnt[p,Qb[b]]
                    if c>=0 and c<best: best=c
            if best<(1<<30): I.append(i); J.append(j); NM.append(best-2); UU.append(u)
    return np.array(I),np.array(J),np.array(NM),np.array(UU)
I,J,NM,UU=spacelike(X.astype(np.float64),Cnt,NPp,NPi,NFp,NFi,ucut,CAP)
print('несравнимых пар у света: %d, %.0f с'%(len(I),time.time()-T0),flush=True)
# калибровка кристалла (разовая): сравнимые n → v; несравнимые n_min → v
cnt_c=[];vol_c=[]
for a in range(0,N,ch):
    b=min(N,a+ch); dt,U=rows(a,b); m=(Cnt[a:b]>=0)&(U<tc2); cnt_c.append(Cnt[a:b][m]); vol_c.append((al*U*U)[m])
cnt_c=np.concatenate(cnt_c).astype(int); vol_c=np.concatenate(vol_c)
vtab=np.array([vol_c[cnt_c==k].mean() if (cnt_c==k).sum()>30 else np.nan for k in range(80)])
for k in range(1,80):
    if not np.isfinite(vtab[k]): vtab[k]=vtab[k-1]+1
vs=al*UU*UU; sel=vs<8; pa,pb=np.polyfit(vs[sel],NM[sel],1)
veff=np.maximum((NM-pb)/pa,0); w_true=np.exp(-vs)*(np.abs(UU)<math.sqrt(tc2)); w_ord=np.exp(-veff)
lam0=w_true.sum()/w_ord.sum(); lam1=(w_true*UU).sum()/(w_ord*UU).sum()
print('калибровка: n→v сравнимые %s…; несравнимые n_min ≈ %.2f·v %+.2f; множитель по M0 %.3f (по M1 был бы %.3f)'%(np.round(vtab[:5],2),pa,pb,lam0,lam1),flush=True)
# ядро по порядку (обход по высоте) и S1 по координатам (обход по времени)
order=np.lexsort((np.arange(N),h)); pos=np.empty(N,np.int64); pos[order]=np.arange(N)
Ko=np.zeros((N,N),np.float32)
for a in range(0,N,ch):
    b=min(N,a+ch); cn=Cnt[a:b]; v=np.where(cn>=0,vtab[np.clip(cn,0,79)],99.0); Ko[a:b]=np.where((cn>=0)&(v<9),0.5*c0*(1-beta)*np.exp(-v),0)
w=0.5*c0*(1+beta)*lam0*w_ord; first=pos[I]<pos[J]; a_=np.where(first,I,J); b_=np.where(first,J,I); Ko[a_,b_]+=w.astype(np.float32)
Ko=Ko[np.ix_(order,order)]; KoT=np.ascontiguousarray(Ko.T); del Ko
KS=np.zeros((N,N),np.float32)
for a in range(0,N,ch):
    b=min(N,a+ch); dt,U=rows(a,b); v=al*U*U; KS[a:b]=np.where((dt>0)&(np.abs(U)<tc2),0.5*c0*np.exp(-v)*np.where(U>0,1-beta,1+beta),0)
KST=np.ascontiguousarray(KS.T); del KS, Cnt
at=Af.Atom(X,rho,M,H,R); mb=at.mb2(M); f=at.src()[:,0].real; Un=-Af.AL/np.sqrt(rr**2+Af.A**2)
def run(KT,Bv,perm):
    ff=f[perm]; BB=Bv[perm]; phi=np.zeros(N); src=ff.copy()
    for j in range(N):
        phi[j]=KT[j,:j]@src[:j]; src[j]+=BB[j]*phi[j]
    out=np.empty(N); out[perm]=phi; return out
def lev(phi,lab):
    if not np.all(np.isfinite(phi)): print('   ',lab,'РАСХОДИТСЯ',flush=True); return np.nan
    Y,_=Af.shell_channels(at,phi.astype(complex)); o,g,w_=Hv.pencil(Y,0.5,8); k=(o>0.05)&(o<M)&(np.abs(g)<0.15)
    print('   ',lab,' '.join('%.3f/%+.3f'%(a,b) for a,b,cw in zip(o,g,w_) if a>=0 and cw>0.03*w_[0])[:100],flush=True)
    return o[np.nonzero(k)[0][0]] if k.any() else np.nan
ident=np.arange(N)
for nm,KT,perm in (('S1 координаты',KST,ident),('порядок',KoT,order)):
    lev(run(KT,np.full(N,-mb),perm),nm+' свободно:')
    E=Af._fixed_point(lambda E: lev(run(KT,-(mb+2*E*Un-Un*Un),perm),nm),0.85*M,tol=5e-4,it=5); print('%s: 1s = %.4f'%(nm,E),flush=True)
print('непрерывная (обрезка): %.4f'%Af.cont_level_hinv(M,R,H),flush=True)
