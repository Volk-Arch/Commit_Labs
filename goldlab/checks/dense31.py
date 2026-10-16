"""3+1 на более плотном кристалле (уйти от зерна: m=1 при ρ=4, m/ρ^¼ = 0.71). Проверки по порядку:
(1) рост отклика со временем (неустойчивость при m ~ ρ^¼); (2) наклон уровня по массе dE/dm²; (3) знак второго порядка
(штарковская вставка z·e^{−r/2}); (4) спиновый сдвиг Дирака против КГ. Прибор тот же, что при ρ=1 (окно, Ханн, вес e^{−αr})."""
import sys, os, math, time, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, propagate as Pr, observe as Ob, causal as Cs
rho=float(sys.argv[1]); H=float(sys.argv[2]); Rc=float(sys.argv[3]); m=1.0; T0=time.time()
lg=lambda *s: print('[%5.0f c]'%(time.time()-T0),*s,flush=True)
w=Wd.gold31(rho,'cylinder',H=H,Rc=Rc,Rs=3.0); X=w.X; cache='links31_rho%g_H%g_R%g.npz'%(rho,H,Rc)
if os.path.exists(cache): z=np.load(cache); I,J=z['I'],z['J']
else:
    I,J=Cs.links_sparse(X,rho,1.6,progress=lambda j,n: lg('связи %d / %d'%(j,n)) if j%20000==0 else None)
    I=I.astype(np.int32); J=J.astype(np.int32); np.savez(cache,I=I,J=J)
lg('мир ρ=%g H=%g R=%g: %d событий, %d связей (%.0f на событие)'%(rho,H,Rc,w.n,len(I),len(I)/w.n))
r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; src0=t<0.45; sel=(t>1.5)&(t<H-0.3); tt=t[sel]
order=np.argsort(J,kind='stable'); I2=I[order]; starts=np.searchsorted(J[order],np.arange(w.n+1)); del J
coef=math.sqrt(rho)/(2*math.pi*math.sqrt(6))
def wv_for(al,a):
    U=lambda q: -al/np.sqrt(q*q+a*a); Jfull=np.repeat(np.arange(w.n),np.diff(starts)).astype(np.int32)
    return coef*Pr.wilson31(X,I2,Jfull,U)          # уже в порядке по J
def run_scalar(wv,Bdiag,prof):
    phi=np.zeros(w.n,complex); src=np.where(src0,prof,0).astype(complex)
    for j in range(w.n):
        s0,s1=starts[j],starts[j+1]
        if s1>s0: phi[j]=(wv[s0:s1]*src[I2[s0:s1]]).sum()
        src[j]+=Bdiag[j]*phi[j]
    return phi
def run_spinor(wv,Bx,prof,e):
    phi=np.zeros((w.n,2),complex); src=np.where(src0[:,None],prof[:,None]*e[None,:],0).astype(complex)
    for j in range(w.n):
        s0,s1=starts[j],starts[j+1]
        if s1>s0: phi[j]=wv[s0:s1]@src[I2[s0:s1]]
        src[j]+=Bx[j]@phi[j]
    return phi@e
def peak(sig,prof,om=np.linspace(0.5,1.3,1601)):
    S=Ob.spectrum(tt,prof[sel]*sig[sel],om,'hann'); pk=Ob.peaks(om,S,0.0); return pk[int(np.argmax([h for _,h in pk]))][0]
def growth(sig,prof):
    e=np.arange(0,H+1e-9,1.0); k=np.digitize(t,e)-1; ok=(k>=0)&(k<len(e)-1)
    amp=np.sqrt(np.bincount(k[ok],np.abs(prof*sig)[ok]**2,minlength=len(e)-1)/np.maximum(np.bincount(k[ok],minlength=len(e)-1),1))
    tc=0.5*(e[1:]+e[:-1]); s=(tc>2)&(tc<H-1); return np.polyfit(tc[s],np.log(amp[s]+1e-300),1)[0]
# --- водород α=0.45, a=0.5 ---
al,a=0.45,0.5; prof=np.exp(-al*r); wv=wv_for(al,a); lg('фазы готовы')
phi=run_scalar(wv,np.full(w.n,-m*m/rho),prof); E1=peak(phi,prof); g1=growth(phi,prof); lg('КГ m²=1: 1s %.5f, темп роста %+.3f'%(E1,g1))
phi=run_scalar(wv,np.full(w.n,-0.81/rho),prof); E2=peak(phi,prof); lg('КГ m²=0.81: 1s %.5f; наклон dE/dm² %.3f'%(E2,(E1-E2)/0.19))
g=X[:,3]*np.exp(-r/2.0); c=0.12/np.abs(g).max()
phi=run_scalar(wv,(-m*m+c*g)/rho,prof); Es=peak(phi,prof); lg('штарк z·e^{−r/2}: сдвиг %+.5f'%(Es-E1))
dU=lambda q: al*q/(q*q+a*a)**1.5
rh=np.where(r[:,None]>1e-9,X[:,1:]/np.maximum(r,1e-9)[:,None],0)
sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]]); sz=np.array([[1,0],[0,-1]],complex)
sr=rh[:,0,None,None]*sx+rh[:,1,None,None]*sy+rh[:,2,None,None]*sz; e=np.array([1,0],complex)
Bx=(-m*m*np.eye(2)[None]+1j*dU(r)[:,None,None]*sr)/rho
Ed=peak(run_spinor(wv,Bx,prof,e),prof); lg('Дирак 1s %.5f; сдвиг Дирак − КГ %+.5f'%(Ed,Ed-E1))
lg('ИТОГ ρ=%g H=%g R=%g: КГ %.5f, рост %+.3f, наклон %.3f, штарк %+.5f, спин %+.5f'%(rho,H,Rc,E1,g1,(E1-E2)/0.19,Es-E1,Ed-E1))
