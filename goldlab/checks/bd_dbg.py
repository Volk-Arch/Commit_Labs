import sys, math, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import atom31f as Af
X=Af.make_world(1.0,14,6); X=X[np.argsort(X[:,0])]; N=len(X); Xf=X.astype(np.float32); t=Xf[:,0]
a,b=6000,6200
dt=t[None,:]-t[a:b,None]; U=dt*dt
for k in (1,2,3): U-=(Xf[None,:,k]-Xf[a:b,None,k])**2
C=(dt>0)&(U>0); print('C rows sum', C.sum(), U.dtype, (U>0).sum(), (dt>0).sum())
Cf=np.zeros((N,N),np.float32)
for a0 in range(0,N,1500):
    dt2=t[None,:]-t[a0:a0+1500,None]; U2=dt2*dt2
    for k in (1,2,3): U2-=(Xf[None,:,k]-Xf[a0:a0+1500,None,k])**2
    Cf[a0:a0+1500]=((dt2>0)&(U2>0))
n=Cf[a:b]@Cf; v=(math.pi/24)*U*U; m=C&(v<16)
print('pairs',m.sum(),' n>0 among them',(n[m]>0).sum(), ' n max', n[m].max())
vv=v[m]; nn=n[m]
for lo,hi in ((0,0.5),(0.5,1.5),(1.5,2.5),(2.5,3.5),(3.5,5)):
    k=(vv>=lo)&(vv<hi); print(lo,hi,k.sum(), nn[k].mean() if k.sum() else None)
