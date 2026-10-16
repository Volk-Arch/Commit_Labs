"""Шаг 2д: атом 3+1 на кристалле с ядром из ПОРЯДКА. Один мир (шар R, окно H), плотные матрицы. Ядра: S1 (координаты, обе стороны),
односторонний по координатам, односторонний по порядку (вес — от числа событий в интервале; только сравнимые пары, без выбора «кто раньше»
у несравнимых). Потенциал — вставкой при энергии E, частота — гармонической инверсией; эталон — непрерывная теория с той же обрезкой."""
import sys, math, time, json, numpy as np
sys.path.insert(0,'/home/claude/goldlab'); sys.path.insert(0,'/home/claude/code/goldlab_checks')
from goldlab import atom31f as Af, hinv as Hv
from onesided_scan import G as Gmean, c as C0, al as AL0, sa as SA0
Af.AL=0.6; M=float(sys.argv[1]); R=float(sys.argv[2]); H=float(sys.argv[3]); bp=float(sys.argv[4]); kind=sys.argv[5] if len(sys.argv)>5 else 'gold'
rho=1.0; beta=Af.BETA; tc2=(2.8*rho**-0.25)**2
if kind=='gold': X=Af.make_world(rho,H,R)
else: X=Af.make_world(rho,H,R,shift='poisson',poisson_seed=int(kind[1:]) if len(kind)>1 else 1)
X=X[np.argsort(X[:,0])]; at=Af.Atom(X,rho,M,H,R); N=len(X); T0=time.time()
import os
CD='/home/claude/scratch_kernels'; names=['S1 (координаты)','одност. (координаты)','одност. (ПОРЯДОК)']
cached=all(os.path.exists('%s/%s_%g_%g_%d.npy'%(CD,kind,R,H,i)) for i in range(3)) and bp==0.05
Xf=X.astype(np.float32); t=Xf[:,0]
def rows(a,b):
    dt=t[None,:]-t[a:b,None]; U=dt*dt
    for k in (1,2,3): U-=(Xf[None,:,k]-Xf[a:b,None,k])**2
    return dt>0, U
if cached: KT={k:np.load('%s/%s_%g_%g_%d.npy'%(CD,kind,R,H,i),mmap_mode='r') for i,k in enumerate(names)}
Cf=np.zeros((N,N),np.float32) if not cached else None
for a in (range(0,N,1500) if not cached else []):
    fu,U=rows(a,min(N,a+1500)); Cf[a:a+1500]=(fu&(U>0))
CfT=None; ch=1500
# таблица n → ρV и ядра — по блокам строк
sv=np.zeros(61); cv=np.zeros(61)
lam=math.sqrt(math.pi)*(1+bp)/(math.pi/2+bp); aa=2*math.sqrt(math.pi)/(math.sqrt(math.pi)-lam); c=C0
def os_w(v): return 0.5*c*aa*np.exp(-v)*(1-lam*np.sqrt(v))
for a in (range(0,N,ch) if not cached else []):
    b=min(N,a+ch); fu,U=rows(a,b); C=fu&(U>0); n=np.minimum(Cf[a:b]@Cf,60).astype(np.int16); vol=(rho*math.pi/24)*U*U
    sv+=np.bincount(n[C],vol[C],61); cv+=np.bincount(n[C],minlength=61)
vtab=np.where(cv>50,sv/np.maximum(cv,1),np.nan) if not cached else np.r_[0.26,0.93,1.57,2.37,3.65,4.41,np.full(55,np.nan)]
for k in range(1,61):
    if not np.isfinite(vtab[k]): vtab[k]=vtab[k-1]+1
print(kind,'N=',N,'матрицы %.0f с; n→ρV:'%(time.time()-T0),' '.join('%.2f'%v for v in vtab[:6]),flush=True)
Kf={k:np.zeros((N,N),np.float32) for k in names} if not cached else {}
for a in (range(0,N,ch) if not cached else []):
    b=min(N,a+ch); fu,U=rows(a,b); C=fu&(U>0); n=np.minimum(Cf[a:b]@Cf,60).astype(np.int16); vol=(rho*math.pi/24)*U*U
    Kf[names[0]][a:b]=np.where(fu&(np.abs(U)<tc2),0.5*c*np.exp(-vol)*np.where(U>0,1-beta,1+beta),0)
    Kf[names[1]][a:b]=np.where(C&(vol<9),os_w(np.minimum(vol,9)),0)
    vn=vtab[n]; Kf[names[2]][a:b]=np.where(C&(vn<9),os_w(np.minimum(vn,9)),0)
if not cached: del Cf
import os
CD='/home/claude/scratch_kernels'; os.makedirs(CD,exist_ok=True); KT={}
for i,k in (enumerate(names) if not cached else []):
    p='%s/%s_%g_%g_%d.npy'%(CD,kind,R,H,i); np.save(p,np.ascontiguousarray(Kf[k].T)); del Kf[k]; KT[k]=np.load(p,mmap_mode='r')
hOS=lambda u: np.where(u>0,os_w(AL0*u*u),0.0)
mb2={'S1 (координаты)':at.mb2(M)}; g=Gmean(hOS,[M])[0][0].real; mb2['одност. (координаты)']=mb2['одност. (ПОРЯДОК)']=M*M/(1-M*M*g)
print('голая масса²:',{k:round(v,4) for k,v in mb2.items()},flush=True)
Un=-Af.AL/np.sqrt(at.r**2+Af.A**2); f=at.src(); e=np.array([1,0])
def run(K,b,pauli):
    Bm=(b[:,None,None]*at.eye+pauli*1j*at.dU[:,None,None]*at.sr)/rho; phi=np.zeros((N,2),complex); s=np.array(f,complex)
    for j in range(N):
        phi[j]=K[j,:j]@s[:j]; s[j]+=Bm[j]@phi[j]
    return phi@e
res={}
def lev(phi):
    if not np.all(np.isfinite(phi)): print('  !! расходится',flush=True); return np.nan
    Y,_=Af.shell_channels(at,phi); o,g,w=Hv.pencil(Y,0.5,8); m=(o>0.05)&(o<M)&(np.abs(g)<0.1)
    print('  моды:',' '.join('%.3f/%+.3f'%(a,b) for a,b in zip(o[:5],g[:5])),flush=True)
    return o[np.nonzero(m)[0][0]] if m.any() else np.nan
ref=Af.cont_level_hinv(M,R,H); refs=Af.cont_level_hinv(M,R,H,pauli=1.0)-ref; refu=Af.cont_level_hinv(M,R,H,trunc=False)
print('непрерывная: обрезка %.4f (без обрезки %.4f), спин %+.5f'%(ref,refu,refs),flush=True)
for nm,K in KT.items():
    m2=mb2[nm]; om=lambda E,p=0.0: lev(run(K,-(m2+2*E*Un-Un*Un),p))
    E=Af._fixed_point(om,0.8*M,tol=3e-4); Es=Af._fixed_point(lambda x: om(x,1.0),E,tol=3e-4)-E
    phi0=run(K,-m2*np.ones(N),0.0); Y,_=Af.shell_channels(at,phi0); o,gm,w=Hv.pencil(Y,0.5,6); i=int(np.argmax(w))
    res[nm]=dict(E=E,spin=Es,free_om=float(abs(o[i])),free_gamma=float(gm[i]))
    print('%-22s 1s %.4f (%+.1f%%)  спин %+.5f (%.0f%%)   свободно: ω=%.3f γ=%+.3f'%(nm,E,100*(E/ref-1),Es,100*Es/refs,abs(o[i]),gm[i]),flush=True)
json.dump(dict(M=M,R=R,H=H,bp=bp,kind=kind,ref=ref,refu=refu,refs=refs,res=res),open('order_atom_%s_M%.3f_bp%.2f.json'%(kind,M,bp),'w'))
