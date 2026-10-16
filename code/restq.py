import sys, numpy as np, math
sys.path.insert(0,'/home/claude/code'); sys.argv=sys.argv[:1]
import chiral as CH
from interact import Rule
w=CH.sym_world(4.0,24.0); X=w.X
for m0 in (0.03, 0.1, 0.3):
    W=Rule(X,m0,0.0)
    s=[k for k in range(W.n) if abs(X[k,1])<1e-9 and -16<X[k,0]<-14][0]
    e0=[e for e in W.eout[s] if abs(W.rap[e])<1e-9][0]
    a,_=W.run(s,e0); t0=X[s,0]
    line=[]; spread=[]
    for dt in np.arange(1,15,1.0):
        t=t0+dt; cr=(X[W.I,0]<t)&(X[W.J,0]>=t)
        xm=X[W.I,1]+(X[W.J,1]-X[W.I,1])*(t-X[W.I,0])/(X[W.J,0]-X[W.I,0])
        p=np.abs(a[cr])**2; P=p.sum(); c=np.abs(xm[cr])<0.6*dt*0+0.8
        line.append(a[cr][c].real.sum()); spread.append(math.sqrt((xm[cr]**2*p).sum()/P)/dt)
    print(f'm0={m0}: разброс |x|/t по срезам: {" ".join(f"{v:.2f}" for v in spread[::2])}; сумма амплитуд у оси по t: {" ".join(f"{v:+.2f}" for v in line)}')
