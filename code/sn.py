"""Различает ли действие конус и ящик: <S>/N против N.

Действие БД в пределе есть Эйнштейн-Гильберт, а для ПЛОСКОГО мира он
нуль. У конуса сумма на событие обязана идти к нулю как N^(-1/4)
(граничный член). У ящика, если он не Минковский, остаться константой.
Если так — действие ОТЛИЧАЕТ их, и exp(-S) выбирает конус.
"""
import math, numpy as np
from core import frac_pairs, rel_of
from weight import w_of

def smeared_S(R, eps):
    A = R.astype(np.float32)
    btw = np.rint(A @ A).astype(np.int32)
    wt = w_of(np.arange(len(R)+2), eps)
    return float(eps*len(R) + wt[np.clip(btw, 0, len(R)+1)][R].sum())

def cone_R(X, chunk=400):
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

EPS, S = 0.2, 6
NS = (400, 800, 1600, 3200)
print(f'eps={EPS}, засевов {S}')
print('предсказание: конус -> 0 как N^(-1/4); ящик -> константа')
print()
print(f"{'мир':>8}{'N':>7}{'доля':>9}{'<S>/N':>12}{'ошибка':>10}")
res={}
for kind in ('конус','ящик'):
    vals=[]
    for N in NS:
        acc=[]
        for t in range(S):
            rs=np.random.default_rng(601000+977*t)
            if kind=='конус': R=cone_R(diamond(N,2.0,rs))
            else: R=rel_of(rs.random((N,4)))
            acc.append((frac_pairs(R), smeared_S(R,EPS)/N)); del R
        A=np.array(acc,float); m=A.mean(0); e=A.std(0)/math.sqrt(S)
        vals.append(m[1])
        print(f'{kind:>8}{N:>7}{m[0]:>9.4f}{m[1]:>12.5f}{e[1]:>10.5f}',
              flush=True)
    res[kind]=np.array(vals)
    p=np.polyfit(np.log(NS), np.log(np.abs(vals)),1)[0]
    print(f'{"показатель по N:":>24} {p:+.3f}   '
          f'(конус должен дать -0.25, ящик 0)')
    print()
