import numpy as np, sys
import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from twobody import build, field_at

def spacing(nbg,k=4):
    rs=np.random.default_rng(1)
    X=rs.random((nbg,k))
    return float(np.mean([np.partition(np.linalg.norm(X-X[i],axis=1),1)[1] for i in range(200)]))

def one(nbg,nsamp,lp,gap,sb,m,k=4,alpha=1.5,spread=0.05):
    rb=np.random.default_rng(sb); rm=np.random.default_rng(sb+11)
    bg=rb.random((nbg,k))
    c1=np.clip(np.full(k,0.5)-gap/2,0.05,0.95); c2=np.clip(np.full(k,0.5)+gap/2,0.05,0.95)
    b1=np.clip(c1+rm.normal(0,spread,(m,k)),0,1); b2=np.clip(c2+rm.normal(0,spread,(m,k)),0,1)
    mid=(c1+c2)/2
    X2,m2,R2,L2,_=build(bg,[b1,b2],k,alpha); f2=field_at(X2,R2,L2,mid,m2,nsamp)
    del X2,R2,L2
    X1,m1,R1,L1,_=build(bg,[b1],k,alpha); f1=field_at(X1,R1,L1,mid,m1,nsamp)
    del X1,R1,L1
    return f2-f1

def const_at(nbg,nsamp,trial,nseed=3,spread=0.05,k=4):
    lp=spacing(nbg,k); gap=4*lp
    pts=[]
    for f in (0.004,0.008,0.012):
        m=max(4,int(f*nbg))
        d=[one(nbg,nsamp,lp,gap,25000+trial*211+s*31,m) for s in range(nseed)]
        d=[x for x in d if np.isfinite(x)]
        if d: pts.append((m*lp**4/spread**4,float(np.mean(d))))
    if len(pts)<3: return np.nan
    s=np.array(pts)
    return abs(np.polyfit(s[:,0],s[:,1],1)[0])
