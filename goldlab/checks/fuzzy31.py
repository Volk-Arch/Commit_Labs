"""Симметрично размытый световой конус в 3+1: пары с t_j > t_i и |τ²| < τ_c² по ОБЕ стороны конуса (сравнимые и несравнимые).
Ядро S1: (c/2)·e^{−α τ⁴}, α = ρπ/24 — гладкое, без связей. Ядро S2: связи с весом c/2 + несравнимые почти световые (c/2)·e^{−ατ⁴}.
Замер: рост, уровень 1s, наклон по массе, второй порядок, спин — против связей и непрерывной теории тем же прибором."""
import sys, math, time, numpy as np, numba
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd, causal as Cs, propagate as Pr, observe as Ob
@numba.njit
def pairs(X, tc2):
    n=X.shape[0]; cap=1<<24; I=np.empty(cap,np.int32); J=np.empty(cap,np.int32); U=np.empty(cap,np.float32); m=0
    for j in range(1,n):
        for i in range(j):
            dt=X[j,0]-X[i,0]
            if dt<=0: continue
            u=dt*dt-(X[j,1]-X[i,1])**2-(X[j,2]-X[i,2])**2-(X[j,3]-X[i,3])**2
            if u<tc2 and u>-tc2:
                if m>=I.shape[0]:
                    I2=np.empty(2*I.shape[0],np.int32); I2[:m]=I[:m]; I=I2
                    J2=np.empty(2*J.shape[0],np.int32); J2[:m]=J[:m]; J=J2
                    U2=np.empty(2*U.shape[0],np.float32); U2[:m]=U[:m]; U=U2
                I[m]=i; J[m]=j; U[m]=u; m+=1
    return I[:m],J[:m],U[:m]
if __name__=='__main__':
    rho=float(sys.argv[1]); H=float(sys.argv[2]); Rc=float(sys.argv[3]); Rs=float(sys.argv[4]); REF=dict(zip(('E','slope','stark','spin'),map(float,sys.argv[5].split(','))))
    al,a,m=0.45,0.5,1.0; alpha=rho*math.pi/24
    w=Wd.gold31(rho,'cylinder',H=H,Rc=Rc,Rs=Rs); X=np.ascontiguousarray(w.X); n=len(X); T0=time.time()
    I,J,U=pairs(X,(2.8*rho**-0.25)**2); U=U.astype(float); print('ρ=%g H=%g R=%g Rs=%g: событий %d, пар %d (%.0f с)'%(rho,H,Rc,Rs,n,len(I),time.time()-T0),flush=True)
    Il,Jl,Nl=Cs.layers31(X,rho,0,2.6)                                  # связи
    key=lambda a_,b_: a_.astype(np.int64)*n+b_
    lk=np.isin(key(I,J),key(Il,Jl))
    r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; prof=np.exp(-al*r); sel=(t>1.5)&(t<H-0.3); src0=t<0.45
    sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]]); sz=np.array([[1,0],[0,-1]],complex)
    rh=np.where(r[:,None]>1e-9,X[:,1:]/np.maximum(r,1e-9)[:,None],0); srr=rh[:,0,None,None]*sx+rh[:,1,None,None]*sy+rh[:,2,None,None]*sz
    dU=al*r/(r*r+a*a)**1.5; eye=np.broadcast_to(np.eye(2),(n,2,2)); g=X[:,3]*np.exp(-r/2.0); cst=0.12/np.abs(g).max(); e=np.array([1,0],complex)
    om=np.linspace(0.5,1.3,1601)
    KER={'связи':(Il,Jl,np.ones(len(Il))),
         'S1 гладкое симметричное':(I,J,0.5*np.exp(-alpha*U*U)),
         }
    for nm,(II,JJ,WW) in KER.items():
        ph=WW*Pr.wilson31(X,II,JJ,lambda q: -al/np.sqrt(q*q+a*a)); f=np.where(src0[:,None],prof[:,None]*e[None,:],0)
        def go(Bx):
            s=Pr.johnston31_respond_spinor(n,II,JJ,rho,m,f,ph,Bx)@e; S=Ob.spectrum(t[sel],prof[sel]*s[sel],om,'hann'); pk=Ob.peaks(om,S,0.0); return pk[int(np.argmax([h for _,h in pk]))][0], s
        E1,s1=go(-m*m/rho*eye); E2,_=go(-0.81/rho*eye)
        ed=np.arange(0,H+1e-9,1.0); kk=np.digitize(t,ed)-1; ok=(kk>=0)&(kk<len(ed)-1); cnt=np.bincount(kk[ok],minlength=len(ed)-1)
        amp=np.sqrt(np.bincount(kk[ok],np.abs(prof*s1)[ok]**2,minlength=len(ed)-1)/np.maximum(cnt,1)); tc_=0.5*(ed[1:]+ed[:-1]); ss=(tc_>2)&(tc_<H-1)&(cnt>0)
        gr=np.polyfit(tc_[ss],np.log(amp[ss]+1e-300),1)[0]
        Es,_=go(((-m*m+cst*g)/rho)[:,None,None]*eye); Ed,_=go((-m*m*eye+1j*dU[:,None,None]*srr)/rho)
        print('  %-26s пар %9d: 1s %.4f (%.4f), рост %+.3f, наклон %.3f (%.3f), штарк %+.5f (%+.5f), спин %+.5f (%+.5f)'%(nm,len(II),E1,REF['E'],gr,(E1-E2)/0.19,REF['slope'],Es-E1,REF['stark'],Ed-E1,REF['spin']),flush=True)
