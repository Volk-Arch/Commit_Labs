"""Насколько воспроизводим спектральный различитель. Эталоны, 16 засевов."""
import math
import numpy as np
from core import spectral, frac_pairs, rel_of

def diamond_cone(N, rs):
    out=[];got=0
    while got<N:
        m=int((N-got)*9)+64
        Y=np.empty((m,4)); Y[:,0]=rs.uniform(0,2,m)
        Y[:,1:]=rs.uniform(-1,1,(m,3))
        r=np.linalg.norm(Y[:,1:],axis=1)
        ok=(Y[:,0]>=r)&(2-Y[:,0]>=r)
        if ok.any(): out.append(Y[ok]); got+=int(ok.sum())
    X=np.vstack(out)[:N]
    dt=X[:,None,0]-X[None,:,0]
    dx=np.linalg.norm(X[:,None,1:]-X[None,:,1:],axis=-1)
    R=dt>dx; np.fill_diagonal(R,False); return np.asarray(R,bool)

def box(N, rs):
    return rel_of(rs.random((N,4)))

S=16
print(f'воспроизводимость spectral(), {S} засевов на точку')
print('если прибор точен, разброс должен быть много меньше '
      'разницы конус-ящик (~0.025)')
print()
print(f"{'мир':>8}{'N':>7}{'доля':>9}{'<спектр>':>11}{'с.к.о.':>9}"
      f"{'ошибка ср.':>12}{'мин':>9}{'макс':>9}")
for name, fn in (('конус', diamond_cone), ('ящик', box)):
    for N in (400, 800, 1600):
        v=[]; f=[]
        for t in range(S):
            rs=np.random.default_rng(12000+911*t)
            R=fn(N,rs); v.append(spectral(R)); f.append(frac_pairs(R)); del R
        v=np.array(v)
        print(f'{name:>8}{N:>7}{np.mean(f):>9.4f}{v.mean():>+11.4f}'
              f'{v.std():>9.4f}{v.std()/math.sqrt(S):>12.4f}'
              f'{v.min():>+9.4f}{v.max():>+9.4f}')
