"""Порядок из беспорядка: выбор делают флуктуации, а не действие.

Если действие вырождено или почти, выигрывает та конфигурация, вокруг
которой ландшафт ПЛОЩЕ: у неё больше флуктуационной энтропии, и свободная
энергия F = <S> - T*ln(объём близких конфигураций) ниже.

Мерим для конуса и ящика при совпадающей плотности:
  <S>/N          — само действие
  с.к.о. dS      — шероховатость ландшафта при переносе одного события
Площе ландшафт => больше энтропии => выигрыш по F.
"""
import math, numpy as np
from core import frac_pairs, mm_dim, rel_of
from weight import w_of

def smeared_S(R, eps):
    A = R.astype(np.float32)
    btw = np.rint(A @ A).astype(np.int32)
    wt = w_of(np.arange(len(R)+2), eps)
    return float(eps*len(R) + wt[np.clip(btw, 0, len(R)+1)][R].sum())

def cone_R(X):
    dt = X[:,None,0]-X[None,:,0]
    dx = np.linalg.norm(X[:,None,1:]-X[None,:,1:], axis=-1)
    R = dt > dx; np.fill_diagonal(R, False); return np.asarray(R, bool)

def diamond(N, T, rs):
    out, got = [], 0
    while got < N:
        m = int((N-got)*9)+64
        Y = np.empty((m,4)); Y[:,0]=rs.uniform(0,T,m)
        Y[:,1:]=rs.uniform(-T/2,T/2,(m,3))
        r=np.linalg.norm(Y[:,1:],axis=1)
        ok=(Y[:,0]>=r)&(T-Y[:,0]>=r)
        if ok.any(): out.append(Y[ok]); got+=int(ok.sum())
    return np.vstack(out)[:N]

N, EPS, MOVES, S = 700, 0.2, 60, 5
print(f'N={N}, eps={EPS}, переносов на мир {MOVES}, засевов {S}')
print('площе ландшафт (меньше с.к.о. dS) => больше энтропии => '
      'ниже свободная энергия')
print()
print(f"{'мир':>10}{'доля':>9}{'разм.':>7}{'<S>/N':>10}{'<|dS|>':>10}"
      f"{'с.к.о. dS':>12}")
for kind in ('конус', 'ящик'):
    rows=[]
    for t in range(S):
        rs = np.random.default_rng(501000+977*t)
        if kind=='конус':
            X = diamond(N, 2.0, rs); R = cone_R(X)
            move = lambda i, r: (lambda Y: (Y.__setitem__(i,
                     diamond(1, 2.0, r)[0]), Y)[1])(X.copy())
            newR = lambda Y: cone_R(Y)
        else:
            X = rs.random((N,4)); R = rel_of(X)
            move = lambda i, r: (lambda Y: (Y.__setitem__(i,
                     r.random(4)), Y)[1])(X.copy())
            newR = lambda Y: rel_of(Y)
        S0 = smeared_S(R, EPS)
        ds=[]
        for _ in range(MOVES):
            i = int(rs.integers(N))
            Y = move(i, rs)
            ds.append(smeared_S(newR(Y), EPS) - S0)
        ds=np.array(ds)
        rows.append((frac_pairs(R), mm_dim(frac_pairs(R)), S0/N,
                     np.abs(ds).mean(), ds.std()))
        del R, X
    A=np.array(rows,float); m=A.mean(0); e=A.std(0)/math.sqrt(S)
    print(f'{kind:>10}{m[0]:>9.4f}{m[1]:>7.2f}{m[2]:>10.4f}'
          f'{m[3]:>10.3f}{m[4]:>12.3f}')
    print(f"{'± ошибка':>10}{e[0]:>9.4f}{e[1]:>7.2f}{e[2]:>10.4f}"
          f"{e[3]:>10.3f}{e[4]:>12.3f}")
