"""ПРАВИЛЬНЫЙ контроль: случайное директорное поле на тех же петлях.

Квантованность кратными 1/2 автоматична для ЛЮБОГО директорного поля.
Содержательна только структура ПЛАТО: сидит ли заряд в конкретных местах.
Контроль — тот же обход, но ось события заменена фиксированным случайным
направлением (поле, а не шум: ось у события одна и та же во всех петлях).
"""
import math, numpy as np
from topo import diamond, cone_rel, axis_of, plane_of

RADII=(0.05,0.08,0.12,0.17,0.23); N,T,S,LOOPS=3000,2.0,3,160

def idx_of(X, R, i0, j0, u, v, rad, cand, field, minsize=25, nst=24,
           pmin=0.25):
    seq=[]
    for th in 2*np.pi*np.arange(nst)/nst:
        tgt=X[j0]+rad*(math.cos(th)*u+math.sin(th)*v)
        d=np.linalg.norm(X[cand]-tgt,axis=1); got=None
        for ii in np.argsort(d)[:14]:
            j=int(cand[ii]); ax=field(j)
            if ax is None: continue
            p=np.array([ax@u, ax@v])
            if np.linalg.norm(p)>=pmin: got=math.atan2(p[1],p[0]); break
        if got is None: return None
        seq.append(got)
    tot=0.0
    for t in range(nst):
        dth=seq[(t+1)%nst]-seq[t]
        tot += (dth+math.pi/2)%math.pi - math.pi/2
    return tot/(2*math.pi)

def run(kind):
    seqs=[]
    for t in range(S):
        rs=np.random.default_rng(221000+977*t)
        X=diamond(N,T,rs); R=cone_rel(X)
        fut=R.sum(0); i0=int(np.argmax(fut*(X[:,0]<0.25*T)))
        cand=np.flatnonzero(R[:,i0])
        if len(cand)<300: continue
        if kind=='real':
            cache={}
            def field(j, _c=cache, _X=X, _R=R, _i=i0):
                if j not in _c: _c[j]=axis_of(_X,_R,_i,j,25)
                return _c[j]
        else:
            rr=np.random.default_rng(9000+t)
            A=rr.normal(size=(len(X),4)); A/=np.linalg.norm(A,axis=1,keepdims=True)
            def field(j, _A=A): return _A[j]
        rs2=np.random.default_rng(600+t)
        for j0 in rs2.choice(cand, min(LOOPS,len(cand)), replace=False):
            j0=int(j0)
            try: u,v=plane_of(X,i0,j0)
            except Exception: continue
            row=[idx_of(X,R,i0,j0,u,v,rad,cand,field) for rad in RADII]
            if all(x is not None for x in row): seqs.append(row)
        del R,X
    return np.array(seqs)

print(f'N={N}, радиусы {RADII}, засевов {S}')
print()
print(f"{'поле':>12}{'посл.':>7}{'|инд|>0.25 при 0.23':>21}"
      f"{'доля ПЛАТО':>12}{'|шаг| средн.':>14}")
for kind,lbl in (('real','настоящее'),('rand','случайное')):
    A=run(kind)
    if len(A)==0: print(f'{lbl:>12}  не построилось'); continue
    D=np.diff(A,axis=1).ravel()
    print(f'{lbl:>12}{len(A):>7}{(np.abs(A[:,-1])>0.25).mean():>21.4f}'
          f'{(np.abs(D)<0.08).mean():>12.4f}{np.abs(D).mean():>14.4f}')
