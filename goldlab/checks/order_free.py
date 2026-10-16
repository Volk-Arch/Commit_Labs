"""Шаг 2г: свободная частица на кристалле — три ядра на одном мире: S1 (координаты, обе стороны), односторонний по координатам,
односторонний по ПОРЯДКУ (вес — от числа событий в интервале). Плотная матрица, мир ~10⁴ событий."""
import sys, math, time, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd, hinv as Hv
kind=sys.argv[1]; Rc=float(sys.argv[2]); H=float(sys.argv[3]); masses=[float(x) for x in sys.argv[4].split(',')]; bp=float(sys.argv[5]) if len(sys.argv)>5 else 0.05
rho=1.0; beta=0.115; c=math.sqrt(rho)/(2*math.pi*math.sqrt(6)); al=rho*math.pi/24; sa=math.sqrt(al); tc2=(2.8*rho**-0.25)**2
if kind=='gold': X=Wd.gold31(rho,'cylinder',H=H,Rc=Rc,Rs=99.0).X
else:
    rng=np.random.default_rng(int(kind[1:]) if len(kind)>1 else 1); n=rng.poisson(rho*(H+0.6)*(2*Rc)**3); P=np.c_[rng.uniform(-0.6,H,n),rng.uniform(-Rc,Rc,(n,3))]
    X=P[np.linalg.norm(P[:,1:],axis=1)<=Rc]
X=X[np.argsort(X[:,0])].astype(np.float32); N=len(X); t=X[:,0]; r=np.linalg.norm(X[:,1:],axis=1); T0=time.time()
dt=t[None,:]-t[:,None]; U=dt*dt
for a in (1,2,3): U-=(X[None,:,a]-X[:,None,a])**2
fut=dt>0; del dt
C=fut&(U>0); Cf=C.astype(np.float32); Cnt=Cf@Cf; del Cf
print(kind,'N=',N,'матрицы %.0f с'%(time.time()-T0),flush=True)
# таблица n → ρV из сравнимых пар этого мира (единственная калибровка; у посыпки по Пуассону было бы n + 1)
vol=(rho*math.pi/24)*U*U; vtab=np.zeros(61)
for n in range(61):
    k=C&(Cnt==n); vtab[n]=vol[k].mean() if k.sum()>50 else vtab[n-1]+1
print('n→ρV:',' '.join('%.2f'%v for v in vtab[:8]),flush=True)
lam=math.sqrt(math.pi)*(1+bp)/(math.pi/2+bp); aa=2*math.sqrt(math.pi)/(math.sqrt(math.pi)-lam)
def os_w(v): s=np.sqrt(v); return 0.5*c*aa*np.exp(-v)*(1-lam*s)
K={}
K['S1']=np.where(fut&(np.abs(U)<tc2),0.5*c*np.exp(-vol)*np.where(U>0,1-beta,1+beta),0).astype(np.float32)
K['одност.коорд']=np.where(C&(vol<9),os_w(vol),0).astype(np.float32)
vn=vtab[np.minimum(Cnt,60).astype(int)]; K['одност.порядок']=np.where(C&(vn<9),os_w(vn),0).astype(np.float32); del vn
K['связи(Джонстон)']=np.where(C&(Cnt==0),c,0).astype(np.float32)
del U,vol,Cnt,C,fut
# голая масса из средней теории каждого ядра (ReG при ω = m), считано onesided_scan.py
src=np.exp(-r*r/4)*(t<0.5)
edges=np.arange(0.5,H-0.2,0.5); kk=np.digitize(t,edges)-1; shells=[(0,1.2),(1.2,2.4),(2.4,3.6)]
def series(phi):
    Y=[]
    for lo,hi in shells:
        m=(r>=lo)&(r<hi)&(kk>=0)&(kk<len(edges)-1); s=np.bincount(kk[m],phi[m],len(edges)-1); n=np.bincount(kk[m],minlength=len(edges)-1)
        Y.append(s/np.maximum(n,1)*(hi**3-lo**3))
    return np.array(Y)
for m in masses:
    for nm,Kx in K.items():
        KT=np.ascontiguousarray(Kx.T); B=-m*m/rho; phi=np.zeros(N); s=src.copy()
        for j in range(N):
            phi[j]=KT[j,:j]@s[:j]; s[j]+=B*phi[j]
        Y=series(phi); om,ga,w=Hv.pencil(Y.astype(complex),0.5,6)
        amp=np.abs(Y).max(0); tc=0.5*(edges[1:]+edges[:-1]); gr=np.polyfit(tc[tc>3],np.log(amp[tc>3]+1e-30),1)[0]
        modes=' '.join('ω=%.3f γ=%+.3f'%(o,g) for o,g,ww in zip(om,ga,w) if 0.05<abs(o)<2.0 and ww>0.05*w[0])
        print('m=%.3f %-16s рост %+.3f | %s'%(m,nm,gr,modes),flush=True)
