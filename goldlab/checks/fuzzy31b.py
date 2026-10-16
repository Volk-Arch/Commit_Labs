"""Размытый конус, экономно по памяти (numba): пары по обе стороны конуса, фаза Вильсона и спинорный отклик — на лету.
Ядра: связи; S1 = (c/2)e^{−αu²}; S3 = (c/2)(3/2 − αu²)e^{−αu²} (без δ″: второй момент по u = τ² нулевой)."""
import sys, math, time, numpy as np, numba
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd, causal as Cs, observe as Ob
from fuzzy31 import pairs
GX,GW=np.polynomial.legendre.leggauss(4)
@numba.njit
def pairs_w(X, tc2, alpha, shape):
    n=X.shape[0]; cap=1<<24; I=np.empty(cap,np.int32); J=np.empty(cap,np.int32); W=np.empty(cap,np.float32); m=0
    for j in range(1,n):
        for i in range(j):
            dt=X[j,0]-X[i,0]
            if dt<=0: continue
            u=dt*dt-(X[j,1]-X[i,1])**2-(X[j,2]-X[i,2])**2-(X[j,3]-X[i,3])**2
            if u<tc2 and u>-tc2:
                if m>=I.shape[0]:
                    cap2=I.shape[0]+(I.shape[0]>>1)
                    I2=np.empty(cap2,np.int32); I2[:m]=I[:m]; I=I2
                    J2=np.empty(cap2,np.int32); J2[:m]=J[:m]; J=J2
                    W2=np.empty(cap2,np.float32); W2[:m]=W[:m]; W=W2
                x=alpha*u*u; I[m]=i; J[m]=j; W[m]=0.5*math.exp(-x) if shape==1 else 0.5*(1.5-x)*math.exp(-x); m+=1
    return I[:m].copy(),J[:m].copy(),W[:m].copy()
@numba.njit
def wilson(X,I,J,al,a,gx,gw):
    out=np.empty(len(I),np.complex64)
    for p in range(len(I)):
        i=I[p]; j=J[p]; s=0.0
        for q in range(4):
            f=0.5*(gx[q]+1); x=X[i,1]+f*(X[j,1]-X[i,1]); y=X[i,2]+f*(X[j,2]-X[i,2]); z=X[i,3]+f*(X[j,3]-X[i,3])
            s+=0.5*gw[q]*(-al/math.sqrt(x*x+y*y+z*z+a*a))
        out[p]=complex(math.cos(s*(X[j,0]-X[i,0])),math.sin(s*(X[j,0]-X[i,0])))
    return out
@numba.njit
def respond(n,I,starts,W,coef,f,B):
    phi=np.zeros((n,2),np.complex128); src=f.copy()
    for j in range(n):
        a0=0j; a1=0j
        for p in range(starts[j],starts[j+1]):
            i=I[p]; w=W[p]; a0+=w*src[i,0]; a1+=w*src[i,1]
        phi[j,0]=coef*a0; phi[j,1]=coef*a1
        src[j,0]+=B[j,0,0]*phi[j,0]+B[j,0,1]*phi[j,1]; src[j,1]+=B[j,1,0]*phi[j,0]+B[j,1,1]*phi[j,1]
    return phi
if __name__=='__main__':
    rho=float(sys.argv[1]); H=float(sys.argv[2]); Rc=float(sys.argv[3]); Rs=float(sys.argv[4]); REF=dict(zip(('E','slope','stark','spin'),map(float,sys.argv[5].split(','))))
    kers=sys.argv[6].split(',') if len(sys.argv)>6 else ['связи','S1','S3']; tcf=float(sys.argv[7]) if len(sys.argv)>7 else 2.6
    al,a,m=0.45,0.5,1.0; alpha=rho*math.pi/24; coef=math.sqrt(rho)/(2*math.pi*math.sqrt(6))
    w=Wd.gold31(rho,'cylinder',H=H,Rc=Rc,Rs=Rs); X=np.ascontiguousarray(w.X); n=len(X); T0=time.time()
    r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; prof=np.exp(-al*r); sel=(t>1.5)&(t<H-0.3); src0=t<0.45
    sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]]); sz=np.array([[1,0],[0,-1]],complex)
    rh=np.where(r[:,None]>1e-9,X[:,1:]/np.maximum(r,1e-9)[:,None],0); srr=rh[:,0,None,None]*sx+rh[:,1,None,None]*sy+rh[:,2,None,None]*sz
    dU=al*r/(r*r+a*a)**1.5; eye=np.broadcast_to(np.eye(2),(n,2,2)); g=X[:,3]*np.exp(-r/2.0); cst=0.12/np.abs(g).max(); e=np.array([1,0],complex)
    f=np.ascontiguousarray(np.where(src0[:,None],prof[:,None]*e[None,:],0).astype(np.complex128)); om=np.linspace(0.5,1.3,1601)
    Bs={'m1':np.ascontiguousarray((-m*m/rho)*eye,dtype=np.complex128),'m081':np.ascontiguousarray((-0.81/rho)*eye,dtype=np.complex128),
        'st':np.ascontiguousarray(((-m*m+cst*g)/rho)[:,None,None]*eye,dtype=np.complex128),'sp':np.ascontiguousarray((-m*m*eye+1j*dU[:,None,None]*srr)/rho,dtype=np.complex128)}
    out={}
    for nm in kers:
        if nm=='связи': I,J,_=Cs.layers31(X,rho,0,2.6); W=np.ones(len(I),np.complex64)
        else:
            I,J,W=pairs_w(X,(tcf*rho**-0.25)**2,alpha,1 if nm=='S1' else 3); W=W.astype(np.complex64)
        W*=wilson(X,I,J,al,a,GX,GW); starts=np.searchsorted(J,np.arange(n+1)).astype(np.int64)
        def go(B):
            s=respond(n,I,starts,W,coef,f,B)@e; S=Ob.spectrum(t[sel],prof[sel]*s[sel],om,'hann'); pk=Ob.peaks(om,S,0.0); return pk[int(np.argmax([h for _,h in pk]))][0], s
        E1,s1=go(Bs['m1']); E2,_=go(Bs['m081']); Es,_=go(Bs['st']); Ed,_=go(Bs['sp'])
        ed=np.arange(0,H+1e-9,1.0); kk=np.digitize(t,ed)-1; ok=(kk>=0)&(kk<len(ed)-1); cnt=np.bincount(kk[ok],minlength=len(ed)-1)
        amp=np.sqrt(np.bincount(kk[ok],np.abs(prof*s1)[ok]**2,minlength=len(ed)-1)/np.maximum(cnt,1)); tc_=0.5*(ed[1:]+ed[:-1]); ss=(tc_>2)&(tc_<H-1)&(cnt>0)
        gr=np.polyfit(tc_[ss],np.log(amp[ss]+1e-300),1)[0]; out[nm]=dict(E=E1,growth=gr,slope=(E1-E2)/0.19,stark=Es-E1,spin=Ed-E1,pairs=len(I))
        print('[%4.0f c] ρ=%g H=%g R=%g Rs=%g %-6s пар %9d: 1s %.4f (%.4f), рост %+.3f, наклон %.3f (%.3f), штарк %+.5f (%+.5f), спин %+.5f (%+.5f)'%(time.time()-T0,rho,H,Rc,Rs,nm,len(I),E1,REF['E'],gr,(E1-E2)/0.19,REF['slope'],Es-E1,REF['stark'],Ed-E1,REF['spin']),flush=True)
        del I,J,W
