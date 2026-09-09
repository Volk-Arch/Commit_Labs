import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'abs_rule.py'), encoding='utf-8').read())

def order_to_scales(R, k, mode='mixed'):
    """Из порядка достаём k величин, которые снова станут шкалами."""
    Rb = np.ascontiguousarray(np.asarray(R, dtype=bool))
    Rf = Rb.astype(np.float32)
    N = len(Rb)
    past = Rf.sum(axis=1); fut = Rf.sum(axis=0)
    dep = proper_time(Rf).astype(float)
    codep = proper_time(np.ascontiguousarray(Rf.T)).astype(float)
    A = Rf; btw = (A@A)
    L = Rb & (btw == 0)
    lin = L.sum(axis=1).astype(float); lout = L.sum(axis=0).astype(float)
    cand = [past, fut, dep, codep, lin, lout, past-fut, dep-codep]
    if mode == 'spectral':
        # собственные векторы симметризованной матрицы
        S = (Rf + Rf.T)
        w, v = np.linalg.eigh(S.astype(np.float64))
        cand = [v[:, -(i+1)] for i in range(k)]
    out = []
    for c in cand[:k]:
        c = np.asarray(c, dtype=float)
        c = c + 1e-9*np.random.default_rng(0).random(N)   # разрываем ничьи
        out.append(np.argsort(np.argsort(c))/N)
    return np.column_stack(out)

def order_to_scales_pca(R, k, rs=None):
    """Из порядка достаём НЕЗАВИСИМЫЕ направления — главные компоненты
    набора признаков события, а не сами скоррелированные признаки."""
    r = rs if rs is not None else np.random.default_rng(0)
    Rb = np.ascontiguousarray(np.asarray(R, dtype=bool))
    Rf = Rb.astype(np.float32); N = len(Rb)
    past = Rf.sum(axis=1); fut = Rf.sum(axis=0)
    dep = proper_time(Rf).astype(float)
    codep = proper_time(np.ascontiguousarray(Rf.T)).astype(float)
    A = Rf; btw = (A@A); L = Rb & (btw == 0)
    lin = L.sum(axis=1).astype(float); lout = L.sum(axis=0).astype(float)
    F = np.column_stack([past, fut, dep, codep, lin, lout,
                         past*fut, dep*codep, past-fut, dep-codep])
    Z = (F - F.mean(0))/np.maximum(F.std(0), 1e-9)
    C = (Z.T @ Z)/len(Z)
    w, v = np.linalg.eigh(C)
    idx = np.argsort(-w)[:k]
    P = Z @ v[:, idx]                       # k независимых направлений
    out = []
    for m in range(k):
        c = P[:, m] + 1e-9*r.random(N)
        out.append(np.argsort(np.argsort(c))/N)
    return np.column_stack(out), w[np.argsort(-w)][:k]/w.sum()
