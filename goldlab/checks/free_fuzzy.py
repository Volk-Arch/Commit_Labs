import sys, math, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd, observe as Ob
from fuzzy_tilt import respond_fly, GX, GW
rho,R,beta=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]); H=12.0; al=0.45
w=Wd.gold31(rho,'cylinder',H=H,Rc=R,Rs=50.0); X=np.ascontiguousarray(w.X); n=len(X)
r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; prof=np.exp(-al*r); sel=(t>1.5)&(t<H-0.3); e=np.array([1,0],complex)
f=np.ascontiguousarray(np.where((t<0.45)[:,None],prof[:,None]*e[None,:],0).astype(np.complex128)); om=np.linspace(0.3,1.6,2601)
alpha=rho*math.pi/24; coef=math.sqrt(rho)/(2*math.pi*math.sqrt(6)); tc2=(2.8*rho**-0.25)**2
for m2 in (0.49,0.81,1.0):
    B=np.ascontiguousarray(np.broadcast_to(np.eye(2)*(-m2/rho),(n,2,2)),dtype=np.complex128)
    for nm,useU in (('свободно',False),):
        s=respond_fly(X,tc2,alpha,coef,0.0,0.5,GX,GW,f,B,False,beta)@e; S=Ob.spectrum(t[sel],prof[sel]*s[sel],om,'hann'); pk=Ob.peaks(om,S,0.0)
        print('ρ=%g R=%g β=%g m²=%.2f: свободный пик %.4f'%(rho,R,beta,m2,pk[int(np.argmax([h for _,h in pk]))][0]),flush=True)
