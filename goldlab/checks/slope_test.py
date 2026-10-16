# Наклон уровня 1s по массе dE/dm² на кристалле 3+1 при разных плотностях (тот же прибор), против непрерывной КГ
import sys, os, math, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, propagate as Pr, observe as Ob, causal as Cs
rho,H,Rc=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]); al,a=0.45,0.5
w=Wd.gold31(rho,'cylinder',H=H,Rc=Rc,Rs=3.0); X=w.X
cache='links31_rho%g_H%g_R%g.npz'%(rho,H,Rc)
if rho==1 and os.path.exists('links31_H%.0f_R%.0f.npz'%(H,Rc)): cache='links31_H%.0f_R%.0f.npz'%(H,Rc)
if os.path.exists(cache): z=np.load(cache); I,J=z['I'],z['J']
else: I,J=Cs.links_sparse(X,rho,1.6); np.savez(cache,I=I.astype(np.int32),J=J.astype(np.int32))
print('ρ=%g H=%g R=%g: событий %d, связей %d'%(rho,H,Rc,w.n,len(I)),flush=True)
U=lambda q: -al/np.sqrt(q*q+a*a); ph=Pr.wilson31(X,I,J,U)
r=np.linalg.norm(X[:,1:],axis=1); prof=np.exp(-r*al); src0=X[:,0]<0.45; sel=(X[:,0]>1.5)&(X[:,0]<H-0.3); tt=X[sel,0]; om=np.linspace(0.6,1.3,1401)
order=np.argsort(J,kind='stable'); I2=I[order]; J2=J[order]; starts=np.searchsorted(J2,np.arange(w.n+1)); wv=ph[order]*math.sqrt(rho)/(2*math.pi*math.sqrt(6))
del ph
def run(m2):
    phi=np.zeros(w.n,complex); src=np.where(src0,prof,0).astype(complex); b=-m2/rho
    for j in range(w.n):
        s0,s1=starts[j],starts[j+1]
        if s1>s0: phi[j]=(wv[s0:s1]*src[I2[s0:s1]]).sum()
        src[j]+=b*phi[j]
    S=Ob.spectrum(tt,prof[sel]*phi[sel],om,'hann'); pk=Ob.peaks(om,S,0.0); return pk[int(np.argmax([h for _,h in pk]))][0]
E1=run(1.0); E2=run(0.81)
print('ρ=%g: 1s(m²=1) %.5f, 1s(m²=0.81) %.5f, наклон dE/dm² %.3f'%(rho,E1,E2,(E1-E2)/0.19),flush=True)
