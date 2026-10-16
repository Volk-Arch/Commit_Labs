"""Откуда разгон: число цепочек из k связей между двумя событиями против непрерывной теории.
Непрерывно (3+1): c^k·N_k(x,y) = ρ^{k−1}·G_k(τ), G_k = τ^{2k−4}/(π·2^{2k−1}(k−1)!(k−2)!), c = √ρ/(2π√6).
k = 2: N₂ = 3π ≈ 9.42 при любом τ;  k = 3: N₃ = 18.13·√ρ·τ²."""
import sys, math, numpy as np, scipy.sparse as sp
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd, causal as Cs
def world(kind,rho,H,seed=0):
    if kind=='gold': return Wd.gold31(rho,'diamond',H=H).X
    rng=np.random.default_rng(seed); n=rng.poisson(rho*math.pi*H**4/24*1.0); P=[]
    hh=H/2
    while len(P)<n:
        q=rng.uniform(-hh,hh,(4*n,4)); q=q[np.abs(q[:,0])+np.linalg.norm(q[:,1:],axis=1)<=hh]; P.extend(q.tolist())
    P=np.array(P[:n]); return P[np.argsort(P[:,0])]
for kind in ('gold','poisson'):
  for rho,H in ((1.0,10.0),(4.0,7.0)):
    X=world(kind,rho,H); n=len(X); I,J=Cs.links_sparse(X,rho,2.1)
    L=sp.csr_matrix((np.ones(len(I)),(J,I)),shape=(n,n))       # L[j,i] = 1, если связь i → j
    L2=L@L; L3=L2@L
    c=math.sqrt(rho)/(2*math.pi*math.sqrt(6))
    # пары: середина ромба, далёкие от края
    cen=np.nonzero(np.abs(X[:,0])+np.linalg.norm(X[:,1:],axis=1)<0.35*H)[0]
    rng=np.random.default_rng(1); out=[]
    for j in rng.choice(cen,min(300,len(cen)),replace=False):
        d=X[j]-X; tau2=d[:,0]**2-(d[:,1:]**2).sum(1); past=np.nonzero((d[:,0]>0)&(tau2>0))[0]
        past=past[np.abs(X[past,0])+np.linalg.norm(X[past,1:],axis=1)<0.45*H]
        r2=L2.getrow(j).toarray().ravel(); r3=L3.getrow(j).toarray().ravel()
        for i in past: out.append((math.sqrt(tau2[i]),r2[i],r3[i]))
    out=np.array(out); print('%s ρ=%g: событий %d, связей на событие %.0f'%(kind,rho,n,len(I)/n))
    for lo,hi in ((0.5,1),(1,1.5),(1.5,2),(2,3),(3,4)):
        s=(out[:,0]*rho**0.25>=lo)&(out[:,0]*rho**0.25<hi)
        if s.sum()<20: continue
        tm=out[s,0].mean()
        print('   τ·ρ^¼ %.1f–%.1f: N₂ = %.2f (теория 9.42), N₃ = %.1f (теория %.1f), пар %d'%(lo,hi,out[s,1].mean(),out[s,2].mean(),18.13*math.sqrt(rho)*np.mean(out[s,0]**2),s.sum()),flush=True)
