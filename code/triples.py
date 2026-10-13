"""Сравнение ТРОЙКАМИ: распределение типов трёхэлементных подпорядков.

Два пространственноподобных события напрямую не сравнить — цепи между
ними нет. Но можно сравнить ОТНОСИТЕЛЬНО ТРЕТЬЕГО. У трёх событий
конечное число типов расположения, и их распределение — порядковый
инвариант, несущий больше, чем одна доля пар.

  цепь        a<b<c
  вилка вниз  a<b, a<c, b||c   (один снизу, два несравнимых сверху)
  вилка вверх a<c, b<c, a||b
  одна связь  a<b, c несравнимо обоим
  антицепь    все несравнимы

Вопрос: разводит ли это конус и ящик резче, чем доля пар и спектр?
Сравнение при СОВПАДАЮЩЕЙ доле пар (ящик при k=4 даёт 0.127, поэтому
конус берётся в области, подогнанной под ту же долю).
"""
import math, numpy as np
from core import frac_pairs, rel_of

def cone_R(X, chunk=300):
    N=len(X); R=np.empty((N,N),bool)
    for a in range(0,N,chunk):
        b=min(a+chunk,N)
        dt=X[a:b,None,0]-X[None,:,0]
        dx=np.linalg.norm(X[a:b,None,1:]-X[None,:,1:],axis=-1)
        R[a:b]=dt>dx
    np.fill_diagonal(R,False); return R

def diamond(N, T, rs):
    out,got=[],0
    while got<N:
        m=int((N-got)*9)+64
        Y=np.empty((m,4)); Y[:,0]=rs.uniform(0,T,m)
        Y[:,1:]=rs.uniform(-T/2,T/2,(m,3))
        r=np.linalg.norm(Y[:,1:],axis=1)
        ok=(Y[:,0]>=r)&(T-Y[:,0]>=r)
        if ok.any(): out.append(Y[ok]); got+=int(ok.sum())
    return np.vstack(out)[:N]

def triple_types(R):
    """Доли типов среди всех C(N,3) троек."""
    N=len(R); A=R.astype(np.float32)
    inc = ~(R|R.T); np.fill_diagonal(inc, False)
    total = N*(N-1)*(N-2)/6.0
    chains = float((A@A).sum())                  # a<b<c, по одной на тройку
    # вилки: для каждого a — несравнимые пары в его будущем / прошлом
    fork_dn = 0.0; fork_up = 0.0
    for a in range(N):
        f=np.flatnonzero(R[:,a])                 # будущее a
        if len(f)>1: fork_dn += inc[np.ix_(f,f)].sum()/2.0
        p=np.flatnonzero(R[a])                   # прошлое a
        if len(p)>1: fork_up += inc[np.ix_(p,p)].sum()/2.0
    # антицепи: тройки попарно несравнимых
    Ai=inc.astype(np.float32)
    anti = float(np.einsum('ij,jk,ki->', Ai, Ai, Ai))/6.0
    rest = total - chains - fork_dn - fork_up - anti
    return (chains/total, fork_dn/total, fork_up/total,
            anti/total, rest/total)

S=4
print('доли типов троек; сравнение при близкой доле пар')
print()
print(f"{'мир':>10}{'доля пар':>10}{'цепь':>10}{'вилка вниз':>12}"
      f"{'вилка вверх':>13}{'антицепь':>11}{'одна связь':>12}")
for kind in ('конус','ящик'):
    rows=[]
    for t in range(S):
        rs=np.random.default_rng(1001000+977*t)
        if kind=='конус': R=cone_R(diamond(600,2.0,rs))
        else: R=rel_of(rs.random((600,4)))
        rows.append((frac_pairs(R),)+triple_types(R)); del R
    A=np.array(rows,float); m=A.mean(0); e=A.std(0)/math.sqrt(S)
    print(f'{kind:>10}{m[0]:>10.4f}{m[1]:>10.5f}{m[2]:>12.5f}'
          f'{m[3]:>13.5f}{m[4]:>11.5f}{m[5]:>12.5f}')
    print(f"{'± ошибка':>10}{e[0]:>10.4f}{e[1]:>10.5f}{e[2]:>12.5f}"
          f"{e[3]:>13.5f}{e[4]:>11.5f}{e[5]:>12.5f}")
