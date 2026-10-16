"""Ящик □ из 5 слоёв прошлого (веса — обнуление моментов M0..M3, Пуассон: 1, −14.0, 41.2, −44.4, 16.2), местный член α(x) = Σ c_k N_k(x)
(□ константы = 0 в каждом событии). Шар R, окно H, плотные матрицы. Сравнение: S1 (координаты) и непрерывная теория. Смягчение ядра A."""
import sys, math, time, numpy as np
sys.path.insert(0,'/home/claude/goldlab'); sys.path.insert(0,'/home/claude/code/goldlab_checks')
from goldlab import atom31f as Af, hinv as Hv
import bd_mean as BM
Af.AL=0.6; R=float(sys.argv[1]); H=float(sys.argv[2]); M=float(sys.argv[3]); Af.A=float(sys.argv[4]); kind=sys.argv[5] if len(sys.argv)>5 else 'gold'; NK=5
rho=1.0; T0=time.time(); beta=Af.BETA; c0=math.sqrt(rho)/(2*math.pi*math.sqrt(6)); tc2=(2.8*rho**-0.25)**2
X=Af.make_world(rho,H,R) if kind=='gold' else Af.make_world(rho,H,R,poisson_seed=int(kind[1:] or 1)); X=X[np.argsort(X[:,0])]; N=len(X)
Xf=X.astype(np.float32); t=Xf[:,0]
def rows(a,b):
    dt=t[None,:]-t[a:b,None]; U=dt*dt
    for k in (1,2,3): U-=(Xf[None,:,k]-Xf[a:b,None,k])**2
    return dt>0, U
Cf=np.zeros((N,N),np.float32); ch=1500
for a in range(0,N,ch):
    fu,U=rows(a,min(N,a+ch)); Cf[a:a+ch]=(fu&(U>0))
L=np.full((N,N),-1,np.int8); KS=np.zeros((N,N),np.float32)
for a in range(0,N,ch):
    b=min(N,a+ch); fu,U=rows(a,b); C=fu&(U>0); n=(Cf[a:b]@Cf); L[a:b]=np.where(C&(n<NK),n,-1).astype(np.int8)
    v=(rho*math.pi/24)*U*U; KS[a:b]=np.where(fu&(np.abs(U)<tc2),0.5*c0*np.exp(-v)*np.where(U>0,1-beta,1+beta),0)
del Cf
LT=np.ascontiguousarray(L.T); del L; KT=np.ascontiguousarray(KS.T); del KS
P=[BM.Pk_poisson(k) for k in range(NK)]; c=BM.design(P,NK); F0,Fs=BM.Bhat(c,P,np.r_[0.05,M],T=80,N=6000); B=Fs-F0; s=-(B[0].real)/0.05**2; mbB=-float((B[1]/s).real)
at=Af.Atom(X,rho,M,H,R); mbS=at.mb2(M); f=at.src()[:,0].real; Un=-Af.AL/np.sqrt(at.r**2+Af.A**2)
cnt=np.array([[np.count_nonzero(LT[x,:x]==k) for k in range(NK)] for x in range(N)])
Wloc=cnt@c
print(kind,'N=',N,'%.0f с'%(time.time()-T0),' c=',np.round(c,2),' s=%.4f α(среднее)=%.4f, α(x): медиана %.4f [%.4f, %.4f]'%(s,F0,np.median(Wloc),np.percentile(Wloc,5),np.percentile(Wloc,95)),' m_b²: ящик %.4f, S1 %.4f'%(mbB,mbS),flush=True)
def box(Mloc):
    phi=np.zeros(N); cc=np.r_[c,0.0]
    for x in range(N):
        lay=LT[x,:x]; m=lay>=0; acc=float(cc[lay[m]]@phi[:x][m]) if x else 0.0
        phi[x]=(f[x]-acc/s)/(Mloc[x]-Wloc[x]/s)
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
# непрерывная: свободная в шаре и атом (тот же прибор)
def cont_modes(AL):
    keep=Af.AL; Af.AL=AL; 
    hh=0.04; r=hh*np.arange(1,int(max(4*R,40)/hh)+1); Nn=len(r); th=(r<R).astype(float); U=-AL/np.sqrt(r*r+Af.A**2)*th; prof=np.exp(-keep*M*r); m2=M*M*th
    def lap(u): d=np.empty_like(u); d[1:-1]=(u[2:]-2*u[1:-1]+u[:-2])/hh**2; d[0]=(u[1]-2*u[0])/hh**2; d[-1]=(-2*u[-1]+u[-2])/hh**2; return d
    def F(y): u,v_=y; return np.array([v_, 2j*U*v_+U*U*u+lap(u)-m2*u])
    y=np.array([np.zeros(Nn,complex),(r*prof*th).astype(complex)]); tt=0.0; dt=0.01; sh=np.linspace(0,R,7); out=[]; ts=[]
    while tt<H-0.2:
        k1=F(y); k2=F(y+dt/2*k1); k3=F(y+dt/2*k2); k4=F(y+dt*k3); y=y+dt/6*(k1+2*k2+2*k3+k4); tt+=dt
        if abs(tt/0.5-round(tt/0.5))<1e-6: ts.append(tt); out.append([np.sum((r*y[0])[(r>=a)&(r<b)])*hh for a,b in zip(sh[:-1],sh[1:])])
    ts=np.array(ts); Y=np.array(out).T; ss=ts>1.5; Af.AL=keep; o,g,w=Hv.pencil(Y[:,ss],0.5,8)
    return ' '.join('%.3f/%+.3f'%(a,b) for a,b,cw in zip(o,g,w) if abs(a)>=0 and cw>0.03*w[0])[:110]
print('непрерывная свободно:',cont_modes(0.0),flush=True)
print('непрерывная атом: 1s(обрезка) = %.4f'%Af.cont_level_hinv(M,R,H),flush=True)
modes(box(np.full(N,mbB)),'ящик свободно:'); modes(s1(np.full(N,-mbS)),'S1   свободно:')
def lev(phi):
    r=modes(phi,'   E-проход')
    if r is None: return np.nan
    o,g,w=r; k=(o>0.05)&(o<M)&(np.abs(g)<0.15); return o[np.nonzero(k)[0][0]] if k.any() else np.nan
for nm,fn in (('ящик',lambda E: lev(box(mbB+2*E*Un-Un*Un))),('S1',lambda E: lev(s1(-(mbS+2*E*Un-Un*Un))))):
    E=Af._fixed_point(fn,0.85*M,tol=5e-4,it=5); print('%s: 1s = %.4f'%(nm,E),flush=True)
