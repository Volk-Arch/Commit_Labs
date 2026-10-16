"""Атом 3+1 в устойчивом режиме: размытый конус (β = 0.115), голая масса из ядра, лёгкая частица m = 1/τ², ядро α = 0.6, a = 0.5.
На каждом мире ансамбля (сдвиги фазона): КГ 1s, наклон по массе, Дирак (спин-орбита), «гелий» J (поле облака — точное, по нити Вильсона);
на первом мире ещё — обе киральности и спин вниз (вырождение)."""
import sys, math, time, json, numpy as np, numba
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd, fuzzy as Fz
from goldlab.core import r_for_density
TAU=(1+5**.5)/2; m=1/TAU**2; al=0.6; a=0.5; rho=1.0; R=11.0; H=20.0; beta=0.115; nw=int(sys.argv[1]) if len(sys.argv)>1 else 3
REF=json.load(open('atom_fz_ref_R%g_H%g_a%g.json'%(R,H,al))); cl=np.load('atom_fz_cloud_a%g.npz'%al)
rg=np.arange(0,70,0.02); Unuc=-al/np.sqrt(rg*rg+a*a); Ucl=np.interp(rg,cl['r'],cl['Uc'],left=cl['Uc'][0])
GX,GW=np.polynomial.legendre.leggauss(4)
@numba.njit
def respond(X,tc2,alpha,coef,Utab,dr,gx,gw,f,B,beta):
    n=X.shape[0]; phi=np.zeros((n,2),np.complex128); src=f.copy(); nt=Utab.shape[0]
    for j in range(n):
        a0=0j; a1=0j
        for i in range(j):
            dt=X[j,0]-X[i,0]
            if dt<=0: continue
            dx=X[j,1]-X[i,1]; dy=X[j,2]-X[i,2]; dz=X[j,3]-X[i,3]; u=dt*dt-dx*dx-dy*dy-dz*dz
            if u>=tc2 or u<=-tc2: continue
            w=0.5*math.exp(-alpha*u*u)*((1.0-beta) if u>0 else (1.0+beta)); s=0.0
            for q in range(4):
                fq=0.5*(gx[q]+1); x=X[i,1]+fq*dx; y=X[i,2]+fq*dy; z=X[i,3]+fq*dz; rr=math.sqrt(x*x+y*y+z*z)/dr; k=int(rr)
                if k>=nt-1: uu=Utab[nt-1]
                else: uu=Utab[k]+(rr-k)*(Utab[k+1]-Utab[k])
                s+=0.5*gw[q]*uu
            c=w*complex(math.cos(s*dt),math.sin(s*dt)); a0+=c*src[i,0]; a1+=c*src[i,1]
        phi[j,0]=coef*a0; phi[j,1]=coef*a1
        src[j,0]+=B[j,0,0]*phi[j,0]+B[j,0,1]*phi[j,1]; src[j,1]+=B[j,1,0]*phi[j,0]+B[j,1,1]*phi[j,1]
    return phi
mb=(Fz.bare_mass2(rho,beta,m),Fz.bare_mass2(rho,beta,0.9*m)); rng=np.random.default_rng(31)
sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]]); sz=np.array([[1,0],[0,-1]],complex)
alpha=rho*math.pi/24; coef=math.sqrt(rho)/(2*math.pi*math.sqrt(6)); tc2=(2.8*rho**-0.25)**2; lam=REF['lam']; om=np.linspace(0.05,0.8,3001)
out=open('atom_fz_ens.jsonl','a'); T0=time.time()
for k in range(nw):
    sh=None if k==0 else tuple(np.array(rng.normal(size=4))*0+ (lambda v: v*r_for_density(rho,4)*rng.random()**0.25/np.linalg.norm(v))(rng.normal(size=4)))
    w=Wd.gold31(rho,'cylinder',H=H,Rc=R,Rs=50.0,shift=sh); X=np.ascontiguousarray(w.X); n=len(X)
    r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; prof=np.exp(-lam*r); sel=(t>1.5)&(t<H-0.3)
    rh=np.where(r[:,None]>1e-9,X[:,1:]/np.maximum(r,1e-9)[:,None],0); srr=rh[:,0,None,None]*sx+rh[:,1,None,None]*sy+rh[:,2,None,None]*sz
    dU=al*r/(r*r+a*a)**1.5; eye=np.broadcast_to(np.eye(2),(n,2,2)); C=lambda A_: np.ascontiguousarray(A_,dtype=np.complex128)
    def go(B,Utab=Unuc,spin=0,read=0):
        e=np.eye(2)[spin]; f=C(np.where((t<0.45)[:,None],prof[:,None]*e[None,:],0))
        s=respond(X,tc2,alpha,coef,Utab,0.02,GX,GW,f,C(B),beta)[:,read]; S=Fz._spec(t[sel],prof[sel]*s[sel],om); i=int(np.argmax(S*(om>0.08)))
        A_,B_,C_=S[i-1],S[i],S[i+1]; return om[i]+0.5*(A_-C_)/(A_-2*B_+C_)*(om[1]-om[0])
    E1=go(-mb[0]/rho*eye); E2=go(-mb[1]/rho*eye); Ed=go((-mb[0]*eye+1j*dU[:,None,None]*srr)/rho); EJ=go(-mb[0]/rho*eye,Utab=Unuc+Ucl)
    rec=dict(world=k,shift=None if sh is None else list(sh),n=n,E=E1,slope=(E1-E2)/(0.19*m*m),spin=Ed-E1,J=EJ-E1)
    if k==0:
        rec['spin_minus']=go((-mb[0]*eye-1j*dU[:,None,None]*srr)/rho)-E1; rec['spin_down']=go((-mb[0]*eye+1j*dU[:,None,None]*srr)/rho,spin=1,read=1)-E1
    out.write(json.dumps(rec)+'\n'); out.flush()
    print('[%5.0f c] мир %d (n=%d): 1s %.4f (%.4f), наклон %.3f (%.3f), спин %+.5f (%+.5f), J %.4f (%.4f)%s'%(time.time()-T0,k,n,E1,REF['E'],rec['slope'],REF['slope'],rec['spin'],REF['spin'],rec['J'],REF['J'],
          '; киральность − %+.5f, спин ↓ %+.5f'%(rec['spin_minus'],rec['spin_down']) if k==0 else ''),flush=True)
