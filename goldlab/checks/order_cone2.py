"""Шаг 2б: τ² несравнимой пары из счёта по четырёхугольникам p ≺ i,j ≺ q (всё — сравнимые интервалы). Пять интервалов задают τ²(i,j)
с точностью до угла в поперечной плоскости; «в одной плоскости» даёт нижнюю границу; оценка = максимум по (p,q)."""
import sys, math, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd
kind=sys.argv[1]; H=float(sys.argv[2]); marg=float(sys.argv[3]); rho=1.0; tcf=2.8
if kind=='gold': X=Wd.gold31(rho,'diamond',H=H).X
else:
    rng=np.random.default_rng(1); hh=H/2; P=rng.uniform(-hh,hh,(rng.poisson(rho*H**4),4)); X=P[np.abs(P[:,0])+np.linalg.norm(P[:,1:],axis=1)<=hh]
X=X[np.argsort(X[:,0])].astype(np.float32); N=len(X); t=X[:,0]
dt=t[None,:]-t[:,None]; U=dt*dt
for a in (1,2,3): U-=(X[None,:,a]-X[:,None,a])**2
C=(dt>0)&(U>0); del dt; Cf=C.astype(np.float32); Cnt=Cf@Cf; del Cf
# калибровка n → ρV по сравнимым парам этого же мира (порядок + одна таблица)
tc2=(tcf*rho**-0.25)**2; vol=rho*math.pi/24*U*U
nmax=40; vtab=np.zeros(nmax+1)
for n in range(nmax+1):
    k=C&(Cnt==n)
    vtab[n]=vol[k].mean() if k.sum()>20 else (vtab[n-1]+1 if n else 0.2)
def tau2(n): n=np.minimum(n.astype(int),nmax); return np.sqrt(24*vtab[n]/(math.pi*rho))
hh=H/2; inner=np.abs(t)+np.linalg.norm(X[:,1:],axis=1)<=hh-marg; idx=np.where(inner)[0]; rng=np.random.default_rng(5); out=[]
for j in rng.choice(idx,min(len(idx),300),replace=False):
    for i in idx:
        if t[i]>=t[j] or C[i,j]: continue
        u=U[i,j]
        if u<=-tc2: continue
        Pm=np.where(C[:,i]&C[:,j]&(Cnt[:,i]<=10)&(Cnt[:,j]<=10))[0]; Qm=np.where(C[i,:]&C[j,:]&(Cnt[i,:]<=10)&(Cnt[j,:]<=10))[0]
        if len(Pm)==0 or len(Qm)==0: continue
        PQ=C[np.ix_(Pm,Qm)]; pp,qq=np.nonzero(PQ); p=Pm[pp]; q=Qm[qq]
        T2=tau2(Cnt[p,q]); a2=tau2(Cnt[p,i]); b2=tau2(Cnt[p,j]); qa=tau2(Cnt[i,q]); qb=tau2(Cnt[j,q]); T=np.sqrt(T2)
        a0=(a2+T2-qa)/(2*T); b0=(b2+T2-qb)/(2*T); ap=np.sqrt(np.maximum(a0*a0-a2,0)); bp=np.sqrt(np.maximum(b0*b0-b2,0))
        est=(a0-b0)**2-(ap+bp)**2; nmin=Cnt[p,q].min()-2
        out.append((u,est.max(),np.median(est),nmin))
A=np.array(out); np.save('order_cone2_%s.npy'%kind,A); print(kind,'N',N,'пар',len(A))
e=np.array([-7.84,-5,-3,-2,-1,-0.5,0])
for a,b in zip(e[:-1],e[1:]):
    k=(A[:,0]>=a)&(A[:,0]<b)
    if k.sum(): print('τ²∈[%.2f,%.2f): max-оценка %.2f ±%.2f   медиана %.2f ±%.2f  (%d)'%(a,b,A[k,1].mean(),A[k,1].std(),A[k,2].mean(),A[k,2].std(),k.sum()))
print('корреляция τ² с max-оценкой %.3f'%np.corrcoef(A[:,0],A[:,1])[0,1])
