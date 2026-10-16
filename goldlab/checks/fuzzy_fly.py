"""Размытый конус S1 «на лету» (numba, без хранения пар): для каждого события j — все i с t_i < t_j и |τ²| < τ_c²,
вес (c/2)e^{−ατ⁴}·(нить Вильсона). Позволяет большие области (без обрезания атома краем)."""
import sys, math, time, numpy as np, numba
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd, observe as Ob
GX,GW=np.polynomial.legendre.leggauss(4)
@numba.njit(parallel=False)
def respond_fly(X,tc2,alpha,coef,al,a,gx,gw,f,B,use_phase):
    n=X.shape[0]; phi=np.zeros((n,2),np.complex128); src=f.copy()
    for j in range(n):
        a0=0j; a1=0j
        for i in range(j):
            dt=X[j,0]-X[i,0]
            if dt<=0: continue
            dx=X[j,1]-X[i,1]; dy=X[j,2]-X[i,2]; dz=X[j,3]-X[i,3]; u=dt*dt-dx*dx-dy*dy-dz*dz
            if u>=tc2 or u<=-tc2: continue
            w=0.5*math.exp(-alpha*u*u)
            if use_phase:
                s=0.0
                for q in range(4):
                    fq=0.5*(gx[q]+1); x=X[i,1]+fq*dx; y=X[i,2]+fq*dy; z=X[i,3]+fq*dz
                    s+=0.5*gw[q]*(-al/math.sqrt(x*x+y*y+z*z+a*a))
                ph=complex(math.cos(s*dt),math.sin(s*dt)); c=w*ph
            else: c=complex(w,0.0)
            a0+=c*src[i,0]; a1+=c*src[i,1]
        phi[j,0]=coef*a0; phi[j,1]=coef*a1
        src[j,0]+=B[j,0,0]*phi[j,0]+B[j,0,1]*phi[j,1]; src[j,1]+=B[j,1,0]*phi[j,0]+B[j,1,1]*phi[j,1]
    return phi
def measure(rho,H,Rc,Rs,REF,tcf=2.8,shift=None,label='',lam=1.0):
    al,a,m=0.45,0.5,1.0; alpha=lam*rho*math.pi/24; coef=math.sqrt(lam*rho)/(2*math.pi*math.sqrt(6)); tc2=(tcf*(lam*rho)**-0.25)**2
    w=Wd.gold31(rho,'cylinder',H=H,Rc=Rc,Rs=Rs,shift=shift); X=np.ascontiguousarray(w.X); n=len(X); T0=time.time()
    r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; prof=np.exp(-al*r); sel=(t>1.5)&(t<H-0.3); src0=t<0.45
    sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]]); sz=np.array([[1,0],[0,-1]],complex)
    rh=np.where(r[:,None]>1e-9,X[:,1:]/np.maximum(r,1e-9)[:,None],0); srr=rh[:,0,None,None]*sx+rh[:,1,None,None]*sy+rh[:,2,None,None]*sz
    dU=al*r/(r*r+a*a)**1.5; eye=np.broadcast_to(np.eye(2),(n,2,2)); g=X[:,3]*np.exp(-r/2.0); cst=0.12/np.abs(g).max(); e=np.array([1,0],complex)
    f=np.ascontiguousarray(np.where(src0[:,None],prof[:,None]*e[None,:],0).astype(np.complex128)); om=np.linspace(0.5,1.3,1601)
    C=lambda A: np.ascontiguousarray(A,dtype=np.complex128)
    def go(B):
        s=respond_fly(X,tc2,alpha,coef,al,a,GX,GW,f,C(B),True)@e; S=Ob.spectrum(t[sel],prof[sel]*s[sel],om,'hann'); pk=Ob.peaks(om,S,0.0); return pk[int(np.argmax([h for _,h in pk]))][0], s
    E1,s1=go((-m*m/rho)*eye); E2,_=go((-0.81/rho)*eye); Es,_=go(((-m*m+cst*g)/rho)[:,None,None]*eye); Ed,_=go((-m*m*eye+1j*dU[:,None,None]*srr)/rho)
    ed=np.arange(0,H+1e-9,1.0); kk=np.digitize(t,ed)-1; ok=(kk>=0)&(kk<len(ed)-1); cnt=np.bincount(kk[ok],minlength=len(ed)-1)
    amp=np.sqrt(np.bincount(kk[ok],np.abs(prof*s1)[ok]**2,minlength=len(ed)-1)/np.maximum(cnt,1)); tc_=0.5*(ed[1:]+ed[:-1]); ss=(tc_>2)&(tc_<H-1)&(cnt>0)
    gr=np.polyfit(tc_[ss],np.log(amp[ss]+1e-300),1)[0]; res=dict(E=E1,growth=gr,slope=(E1-E2)/0.19,stark=Es-E1,spin=Ed-E1,n=n)
    print('[%4.0f c] %sρ=%g H=%g R=%g Rs=%g S1: n=%d, 1s %.4f (%.4f), рост %+.3f, наклон %.3f (%.3f), штарк %+.5f (%+.5f), спин %+.5f (%+.5f)'%(time.time()-T0,label,rho,H,Rc,Rs,n,E1,REF['E'],gr,(E1-E2)/0.19,REF['slope'],Es-E1,REF['stark'],Ed-E1,REF['spin']),flush=True)
    return res
if __name__=='__main__':
    rho=float(sys.argv[1]); H=float(sys.argv[2]); Rc=float(sys.argv[3]); Rs=float(sys.argv[4]); REF=dict(zip(('E','slope','stark','spin'),map(float,sys.argv[5].split(','))))
    lam=float(sys.argv[6]) if len(sys.argv)>6 else 1.0; measure(rho,H,Rc,Rs,REF,lam=lam,label='λ=%g '%lam)
