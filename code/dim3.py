"""Вторая, независимая оценка размерности — по тройкам-цепям.

Доля связанных ПАР даёт размерность по Мирхейму-Мейеру. Доля троек-цепей
c3 тоже определяется одной размерностью (обе безразмерны), значит даёт
ВТОРУЮ независимую оценку. Критерий: у манифолдоподобного порядка они
обязаны СОВПАСТЬ, у подделки — разойтись.

Калибровка по засевам известной размерности d=2..6, формула не нужна.
"""
import math, numpy as np
from core import frac_pairs, rel_of, mm_dim
import growth2
from growth2 import grow
from refgrow import grow_ref

def diamond_d(N, d, rs, T=2.0):
    out,got=[],0
    while got<N:
        m=int((N-got)*14)+64
        Y=np.empty((m,d)); Y[:,0]=rs.uniform(0,T,m)
        Y[:,1:]=rs.uniform(-T/2,T/2,(m,d-1))
        r=np.linalg.norm(Y[:,1:],axis=1) if d>1 else 0
        ok=(Y[:,0]>=r)&(T-Y[:,0]>=r)
        if ok.any(): out.append(Y[ok]); got+=int(ok.sum())
    return np.vstack(out)[:N]

def cone_R(X, chunk=300):
    N=len(X); R=np.empty((N,N),bool)
    for a in range(0,N,chunk):
        b=min(a+chunk,N)
        dt=X[a:b,None,0]-X[None,:,0]
        dx=(np.linalg.norm(X[a:b,None,1:]-X[None,:,1:],axis=-1)
            if X.shape[1]>1 else 0)
        R[a:b]=dt>dx
    np.fill_diagonal(R,False); return R

def c3_of(R):
    """Доля троек, образующих цепь a<b<c."""
    N=len(R); A=R.astype(np.float32)
    return float((A@A).sum())/(N*(N-1)*(N-2)/6.0)

N, S = 700, 4
print(f'N={N}, засевов {S}. Калибровка c3 по известной размерности:')
print(f"{'d':>4}{'доля пар':>11}{'разм. по парам':>16}{'c3':>11}")
cal=[]
for d in (2,3,4,5,6):
    acc=[]
    for t in range(S):
        rs=np.random.default_rng(1101000+977*t)
        R=cone_R(diamond_d(N,d,rs))
        acc.append((frac_pairs(R), c3_of(R))); del R
    m=np.array(acc,float).mean(0)
    cal.append((d, m[0], m[1]))
    print(f'{d:>4}{m[0]:>11.4f}{mm_dim(m[0]):>16.2f}{m[1]:>11.6f}')

ds=np.array([c[0] for c in cal]); c3s=np.array([c[2] for c in cal])
def dim3(c3):
    """Размерность по c3: интерполяция калибровки в логарифмах."""
    lc=np.log(c3s); o=np.argsort(lc)
    return float(np.interp(np.log(c3), lc[o], ds[o]))

print()
print('ПРОВЕРКА: совпадают ли две оценки')
print(f"{'мир':>18}{'доля пар':>10}{'разм.(пары)':>13}{'c3':>11}"
      f"{'разм.(тройки)':>15}{'расхождение':>13}")
def test(lbl, fn):
    acc=[]
    for t in range(S):
        rs=np.random.default_rng(1201000+977*t)
        R=fn(rs); acc.append((frac_pairs(R), c3_of(R))); del R
    m=np.array(acc,float).mean(0)
    d2=mm_dim(m[0]); d3=dim3(m[1])
    print(f'{lbl:>18}{m[0]:>10.4f}{d2:>13.2f}{m[1]:>11.6f}'
          f'{d3:>15.2f}{d3-d2:>+13.2f}')

test('конус d=4', lambda rs: cone_R(diamond_d(N,4,rs)))
test('ящик k=4', lambda rs: rel_of(rs.random((N,4))))
test('refgrow k=6', lambda rs: grow_ref(N,6,rs,1.0,0.0))
def descg(rs):
    growth2._ALPHA=1.0
    return grow(N,'alpha',0.05,0.0,0.2,8,rs)[0]
test('рост по потомкам', descg)
