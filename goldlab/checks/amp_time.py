import sys, math, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd, causal as Cs, propagate as Pr
rho=2.0; al=0.45
for lab,(H,Rc,Rs) in (('рабочий H12 R5 (конус)',(12.0,5.0,3.0)),('рабочий без конуса H12 R5',(12.0,5.0,50.0)),('широкий H12 R8',(12.0,8.0,50.0))):
    X=Wd.gold31(rho,'cylinder',H=H,Rc=Rc,Rs=Rs).X; I,J=Cs.links_sparse(X,rho,1.6); n=len(X)
    r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; f=np.where((t<0.45)&(r<=3.6),np.exp(-al*r),0)
    phi=Pr.johnston31_respond(n,I,J,rho,1.0,f,None); cen=r<2
    ed=np.arange(0,H+1e-9,1.0); k=np.digitize(t,ed)-1; ok=(k>=0)&(k<len(ed)-1)&cen; cnt=np.bincount(k[ok],minlength=len(ed)-1)
    amp=np.sqrt(np.bincount(k[ok],np.abs(phi)[ok]**2,minlength=len(ed)-1)/np.maximum(cnt,1))
    print('%s (событий %d): |φ| в центре по t=0.5,1.5,…: %s'%(lab,n,' '.join('%.3g'%x for x in amp)),flush=True)
