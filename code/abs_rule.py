import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'movingwaves.py'), encoding='utf-8').read())

def rel_absolute(X, thresh=0.5):
    """Правило, чувствительное к АБСОЛЮТНОЙ фазе, а не только к разности.
    i предшествует j, если по каждой шкале фаза j больше фазы i
    И при этом обе лежат по одну сторону от порога (в одной 'полуволне')."""
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k):
        x = X[:, m]
        less = (x[None, :] > x[:, None])
        samehalf = ((x[None, :] < thresh) == (x[:, None] < thresh))
        R &= (less & samehalf)
    np.fill_diagonal(R, False)
    return R

def rel_window(X, w=0.5):
    """Другой вариант: связь есть, только если ОБЕ фазы попали в окно [0,w)."""
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k):
        x = X[:, m]
        inw = x < w
        R &= ((x[None, :] > x[:, None]) & inw[None, :] & inw[:, None])
    np.fill_diagonal(R, False)
    return R
