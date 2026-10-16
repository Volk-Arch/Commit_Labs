"""Ансамбль атома 3+1 (m = 1/τ²): кристалл со сдвигами фазона и посыпка. Строка JSON на мир."""
import sys, json, time, numpy as np; sys.path.insert(0,'/home/claude/goldlab')
from goldlab import atom31f as Af
from goldlab.core import r_for_density
Af.AL=float(sys.argv[6]) if len(sys.argv)>6 else 0.45; M=float(sys.argv[1]); R=float(sys.argv[2]); H=float(sys.argv[3]); kind=sys.argv[4]; ks=[int(x) for x in sys.argv[5].split(',')]
for k in ks:
    T0=time.time(); rng=np.random.default_rng(1000+k)
    if kind=='gold':
        sh=None if k==0 else tuple(rng.normal(size=4)*0)  # заменим
        if k>0: v=rng.normal(size=4); v*=r_for_density(1.0,4)*rng.random()**0.25/np.linalg.norm(v); sh=tuple(v)
        X=Af.make_world(1.0,H,R,shift=sh)
    else: X=Af.make_world(1.0,H,R,poisson_seed=k+1)
    at=Af.Atom(X,1.0,M,H,R); res=at.measure(); res.update(kind=kind,k=k,n=at.n,sec=time.time()-T0)
    print(json.dumps(res),flush=True)
    with open('atom31f_ens_al%g_M%.3f.jsonl'%(Af.AL,M),'a') as fo: fo.write(json.dumps(res)+'\n')
