import sys, math, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd, causal as Cs, propagate as Pr, observe as Ob
al,a=0.45,0.5; H=12.0
for rho in (1.0,2.0,4.0):
    X=Wd.gold31(rho,'cylinder',H=H,Rc=5.0,Rs=3.0).X; z=np.load('links31_rho%g_H12_R5.npz'%rho); I,J=z['I'],z['J']; n=len(X)
    r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; prof=np.exp(-al*r); f=np.where(t<0.45,prof,0)
    ph=Pr.wilson31(X,I,J,lambda q: -al/np.sqrt(q*q+a*a))
    for lab,p in (('свободно',None),('ядро',ph)):
        phi=Pr.johnston31_respond(n,I,J,rho,1.0,f,p)
        ed=np.arange(0,H+1e-9,1.0); k=np.digitize(t,ed)-1; ok=(k>=0)&(k<len(ed)-1); cnt=np.bincount(k[ok],minlength=len(ed)-1)
        proj=np.abs(np.bincount(k[ok],(prof*phi)[ok].real,minlength=len(ed)-1)+1j*np.bincount(k[ok],(prof*phi)[ok].imag,minlength=len(ed)-1))/np.maximum(cnt,1)
        # частоты рано и поздно
        om=np.linspace(0.6,1.4,801); outs=[]
        for t0,t1 in ((1.5,6.5),(6.5,11.7)):
            s=(t>t0)&(t<t1); S=Ob.spectrum(t[s],prof[s]*phi[s],om,'hann'); pk=sorted(Ob.peaks(om,S,0.1),key=lambda q:-q[1])[:2]; outs.append(' '.join('%.3f(%.2f)'%q for q in pk))
        print('ρ=%g %s: |проекция на 1s| по t: %s; пики рано: %s; поздно: %s'%(rho,lab,' '.join('%.2g'%x for x in proj),outs[0],outs[1]),flush=True)
