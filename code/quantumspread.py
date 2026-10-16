import sys, numpy as np
sys.path.insert(0,'/home/claude/code'); sys.argv=sys.argv[:1]
from closedsym import ClosedSym
S=ClosedSym(match='нет'); X=S.X
ea=np.array([e[0] for e in S.edges]); eb=np.array([e[1] for e in S.edges])
src=int(np.argmin(((X-np.array([-10.,0.]))**2).sum(1)))
_,resk,amp,_=S.run({src:np.array([1+0j])})
for dt in (2,4,6,8):
    t=X[src,0]+dt; cr=(X[ea,0]<t)&(X[eb,0]>=t)
    xm=X[ea,1]+(X[eb,1]-X[ea,1])*(t-X[ea,0])/(X[eb,0]-X[ea,0]); p=np.abs(amp[cr,0])**2; P=p.sum()
    u=np.abs(xm[cr]-X[src,1])/dt
    print(f'Δt={dt}: в мире {P:.3f} (в тени {resk[X[:,0]<t,0].sum():.3f}); доля у светового конуса |x|/t>0.8: {p[u>0.8].sum()/P:.2f}, внутри |x|/t<0.4: {p[u<0.4].sum()/P:.2f}; <|x|/t> = {(p*u).sum()/P:.2f}')
