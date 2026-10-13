import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/loopback.py').read())
from sklearn.ensemble import HistGradientBoostingRegressor as GBR

def order_features(R):
    """Признаки события ИЗ ПОРЯДКА — только они доступны на выходе."""
    Rb = np.ascontiguousarray(np.asarray(R, dtype=bool))
    Rf = Rb.astype(np.float32); N = len(Rb)
    past = Rf.sum(axis=1); fut = Rf.sum(axis=0)
    dep = proper_time(Rf).astype(float)
    codep = proper_time(np.ascontiguousarray(Rf.T)).astype(float)
    A = Rf; btw = (A@A); L = Rb & (btw == 0)
    lin = L.sum(axis=1).astype(float); lout = L.sum(axis=0).astype(float)
    return np.column_stack([past, fut, dep, codep, lin, lout,
                            past/N, fut/N, past*fut/N**2, dep-codep, past-fut])

def r2_of(F, y, seed=0):
    n = len(y); h = n//2
    m = GBR(max_iter=250, random_state=seed).fit(F[:h], y[:h])
    p = m.predict(F[h:]); yt = y[h:]
    return float(1 - np.var(yt-p)/max(np.var(yt), 1e-12))

def symmetric_targets(U):
    """Симметричные функции шкал — кандидаты в восстановимые величины."""
    N, k = U.shape
    S = {}
    S['сумма'] = U.sum(1)
    S['произведение'] = np.prod(U, 1)
    S['сумма квадратов'] = (U**2).sum(1)
    S['максимум'] = U.max(1)
    S['минимум'] = U.min(1)
    S['размах'] = U.max(1) - U.min(1)
    S['дисперсия'] = U.var(1)
    S['медиана'] = np.median(U, 1)
    S['вторая эл. симм.'] = (U.sum(1)**2 - (U**2).sum(1))/2
    S['сумма кубов'] = (U**3).sum(1)
    return S
