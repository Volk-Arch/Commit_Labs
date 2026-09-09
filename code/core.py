import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'scaling.py'), encoding='utf-8').read())
from order8 import sprinkle, order_of, dimension, distinctness

def bundle(N, k, rs):
    X = rs.random((N, k))
    R = rel_of(X); Rb = np.asarray(R, dtype=bool)
    D = proper_time(np.asarray(R, dtype=np.float32))
    A = Rb.astype(np.float32); L = Rb & ((A@A) == 0)
    K = R.T.astype(np.float32)
    f = R.sum()/(N*(N-1)/2)
    d = dimension(K); o = distinctness(K)
    h = int(D.max())
    lk = L.sum()/N
    # доля верхушки общего прошлого и конфликтность
    tops = []; confs = []
    for _ in range(4000):
        i, j = (int(v) for v in rs.integers(0, N, 2))
        if i == j or Rb[i, j] or Rb[j, i]: continue
        m = np.flatnonzero(Rb[i] & Rb[j])
        if len(m) < 10: continue
        sub = Rb[np.ix_(m, m)]
        top = int((~sub.any(axis=0)).sum())
        tops.append(top/len(m))
        oi = int(Rb[i].sum()-len(m)); oj = int(Rb[j].sum()-len(m))
        confs.append(top/max(1, min(oi, oj)))
        if len(tops) >= 200: break
    return dict(доля_пар=f, размерность=d, различимость=o,
                высота_норм=h/N**(1.0/k), связей_на_элемент=lk,
                верхушка=float(np.mean(tops)) if tops else np.nan,
                конфликтность=float(np.mean(confs)) if confs else np.nan)
