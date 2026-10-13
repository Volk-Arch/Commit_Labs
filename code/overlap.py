"""Профиль перекрытия прошлых: несёт ли он отличие конуса от ящика.

|past(x) ∩ past(y)| считается одним умножением: (R @ R.T)[x,y].
Жаккар J = пересечение / объединение, отдельно для связанных и
несравнимых пар. В конусе перекрытие прошлых — пересечение конусов,
жёсткая решётка; в перколяции — случайные подмножества.
"""
import math, numpy as np
from core import rel_of, spectral, frac_pairs, mm_dim
import growth2
from growth2 import grow

N, S = 700, 6

def cone(N, rs):
    out=[];got=0
    while got<N:
        m=int((N-got)*9)+64
        Y=np.empty((m,4)); Y[:,0]=rs.uniform(0,2,m); Y[:,1:]=rs.uniform(-1,1,(m,3))
        r=np.linalg.norm(Y[:,1:],axis=1); ok=(Y[:,0]>=r)&(2-Y[:,0]>=r)
        if ok.any(): out.append(Y[ok]); got+=int(ok.sum())
    X=np.vstack(out)[:N]
    dt=X[:,None,0]-X[None,:,0]
    dx=np.linalg.norm(X[:,None,1:]-X[None,:,1:],axis=-1)
    R=dt>dx; np.fill_diagonal(R,False); return np.asarray(R,bool)

def box(N, rs): return rel_of(rs.random((N,4)))
def perc(N, rs):
    growth2._ALPHA=0.0
    return grow(N,'alpha',0.03,0.0,0.2,8,rs)[0]
def grown(N, rs):
    growth2._ALPHA=1.0
    return grow(N,'alpha',0.05,0.0,0.2,8,rs)[0]

def profile(R):
    A = R.astype(np.float32)
    inter = A @ A.T                     # |past(x) & past(y)|
    sz = A.sum(1)
    union = sz[:,None] + sz[None,:] - inter
    J = np.where(union>0, inter/np.maximum(union,1e-9), 0.0)
    rel = R | R.T
    iu = np.triu(np.ones_like(rel), 1).astype(bool)
    big = (sz[:,None]>=5) & (sz[None,:]>=5)
    m_rel = iu & rel & big
    m_inc = iu & ~rel & big
    if m_rel.sum()<50 or m_inc.sum()<50: return (np.nan,)*5
    jr, ji = J[m_rel], J[m_inc]
    return jr.mean(), ji.mean(), jr.mean()/max(ji.mean(),1e-9), ji.std(), float(m_inc.sum()/iu.sum())

print(f'N={N}, засевов {S}. J = перекрытие прошлых (Жаккар)')
print()
print(f"{'мир':>10}{'доля':>8}{'разм.':>7}{'спектр':>9}"
      f"{'J связ.':>10}{'J несравн.':>12}{'отношение':>11}{'с.к.о. J':>10}")
for name, fn in (('конус',cone), ('ящик',box), ('перколяция',perc),
                 ('рост a=1',grown)):
    acc=[]
    for t in range(S):
        rs=np.random.default_rng(61000+977*t)
        R=fn(N,rs); f=frac_pairs(R)
        acc.append((f, mm_dim(f), spectral(R)) + profile(R)); del R
    A=np.array(acc,float); m=np.nanmean(A,0)
    print(f'{name:>10}{m[0]:>8.4f}{m[1]:>7.2f}{m[2]:>+9.4f}'
          f'{m[3]:>10.4f}{m[4]:>12.4f}{m[5]:>11.2f}{m[6]:>10.4f}')
