"""Шаг 2: можно ли размытый конус S1 считать из порядка. Тождество: ατ⁴ = ρπτ⁴/24 = ρ·V(наименьший ромб, содержащий обе точки)
(для сравнимых — сам интервал, для несравнимых — наименьший ромб I(p,q) ∋ i,j). Проверка: n (счёт событий в этом ромбе, без i,j)
против ρV из координат — на кристалле и на посыпке."""
import sys, math, json, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd
kind=sys.argv[1]; H=float(sys.argv[2]); rho=1.0; tcf=2.8
if kind=='gold':
    X=Wd.gold31(rho,'diamond',H=H).X
else:
    rng=np.random.default_rng(int(sys.argv[3]) if len(sys.argv)>3 else 1); hh=H/2; V=math.pi/24*H**4; nn=rng.poisson(rho*V*16/ (math.pi*4/3*1) )
    P=rng.uniform(-hh,hh,(rng.poisson(rho*H**4),4)); X=P[np.abs(P[:,0])+np.linalg.norm(P[:,1:],axis=1)<=hh]
X=X[np.argsort(X[:,0])]; N=len(X); t=X[:,0]
X=X.astype(np.float32); t=X[:,0]; dt=t[None,:]-t[:,None]; U=dt*dt
for a in (1,2,3): U-= (X[None,:,a]-X[:,None,a])**2   # U[a,b] = τ² от a к b
C=(dt>0)&(U>0); Cf=C.astype(np.float32); Cnt=(Cf@Cf)   # Cnt[a,b] = #{k: a≺k≺b}
del dt
tc2=(tcf*rho**-0.25)**2; hh=H/2; inner=np.abs(t)+np.linalg.norm(X[:,1:],axis=1)<=hh-float(sys.argv[4] if len(sys.argv)>4 else 3.5)
idx=np.where(inner)[0]; rng=np.random.default_rng(5); out=[]
for j in rng.choice(idx,min(len(idx),400),replace=False):
    for i in idx:
        if t[i]>=t[j]: continue
        u=U[i,j]
        if abs(u)>=tc2: continue
        if C[i,j]: n=Cnt[i,j]
        else:
            Pm=np.where(C[:,i]&C[:,j]&(Cnt[:,i]<=12)&(Cnt[:,j]<=12))[0]; Qm=np.where(C[i,:]&C[j,:]&(Cnt[i,:]<=12)&(Cnt[j,:]<=12))[0]
            if len(Pm)==0 or len(Qm)==0: continue
            n=Cnt[np.ix_(Pm,Qm)].min()-2
        out.append((u,rho*math.pi/24*u*u,float(n)))
A=np.array(out); np.save('/home/claude/code/goldlab_checks/order_cone_%s.npy'%kind,A)
print(kind,'N=',N,'пар',len(A))
for side,s in (('сравнимые',A[:,0]>0),('несравнимые',A[:,0]<0)):
    B=A[s]; e=np.array([0,0.25,0.5,1,1.5,2,3,4,6,8.1])
    print(side, 'пар',len(B), ' <e^-n>/<e^-ρV> =%.3f'%(np.exp(-B[:,2]).mean()/np.exp(-B[:,1]).mean()))
    for a,b in zip(e[:-1],e[1:]):
        k=(B[:,1]>=a)&(B[:,1]<b)
        if k.sum(): print('  ρV∈[%.2f,%.2f): n̄=%.2f ±%.2f  (%d)'%(a,b,B[k,2].mean(),B[k,2].std(),k.sum()))
