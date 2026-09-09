import numpy as np, gc, sys
sys.path.insert(0,'/home/claude')
from bigworld import sprinkle, interval_of

def coef(N, lo, hi, want, sd, warps=(0,40,120)):
    pts=[]
    for warp in warps:
        P=sprinkle(N,4,np.random.default_rng(164000+warp*7+sd*211),warp=warp)
        rs=np.random.default_rng(sd*23+warp); rows=[]
        for _ in range(20000):
            i,j=(int(v) for v in rs.integers(0,len(P),2))
            dt=float(P[i,0]-P[j,0])
            if dt<=0: continue
            dx2=float(((P[i,1:]-P[j,1:])**2).sum())
            if dt*dt<=dx2: continue
            mid=interval_of(P,i,j); n=len(mid)
            if n<lo or n>hi:
                del mid; continue
            Q=P[mid]
            a=Q[None,:,0]-Q[:,None,0]; b=((Q[None,:,1:]-Q[:,None,1:])**2).sum(-1)
            Rm=(a>0)&(a*a>b)
            A2=Rm.astype(np.float32); Lm=Rm&((A2@A2)==0)
            tot=float(Rm.sum())
            rows.append((float(Lm.sum())/max(tot,1), tot/(n*(n-1)/2)))
            del mid,Q,Rm,A2,Lm
            if len(rows)>=want: break
        del P; gc.collect()
        if len(rows)<12: return np.nan
        A=np.array(rows)
        pts.append((float(np.mean(A[:,0])), float(np.mean(A[:,1]))))
    B=np.array(pts)
    return float(np.polyfit(B[:,1],B[:,0],1)[0])
