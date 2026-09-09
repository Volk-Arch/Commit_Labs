import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'macro_entropy.py'), encoding='utf-8').read())

def labels_of(X):
    """Метка события: какая шкала выбивается. Вычисляется из фаз."""
    med = np.median(X, axis=1, keepdims=True)
    dev = np.abs(X - med)
    lab = dev.argmax(axis=1)
    srt = np.sort(dev, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    return np.where(margin >= 0.02, lab, -1), margin

def charged_step(X, rate=0.08, mix=0.5, g=0.0, noise=0.002, rs=None):
    """Как раньше, ПЛЮС взаимодействие по метке:
    g>0 — одинаковые метки притягиваются, разные отталкиваются
    g<0 — наоборот. g=0 — прежняя динамика (контроль)."""
    r = rs if rs is not None else np.random.default_rng(0)
    R = rel_of(X); Rf = R.astype(np.float64)
    npast = np.maximum(Rf.sum(axis=1), 1)[:, None]
    nfut = np.maximum(Rf.sum(axis=0), 1)[:, None]
    cp = (Rf @ X)/npast; cf = (Rf.T @ X)/nfut
    attract = 0.5*(cp+cf); repel = 2*X - attract
    target = mix*attract + (1-mix)*repel
    Xn = X + rate*(target - X)
    if g != 0.0:
        lab, _ = labels_of(X)
        same = (lab[None, :] == lab[:, None]) & (lab[None, :] >= 0)
        S = (Rf > 0) & same
        D = (Rf > 0) & ~same & (lab[None, :] >= 0) & (lab[:, None] >= 0)
        ns = np.maximum(S.sum(axis=1), 1)[:, None]
        nd = np.maximum(D.sum(axis=1), 1)[:, None]
        cs = (S.astype(np.float64) @ X)/ns
        cd = (D.astype(np.float64) @ X)/nd
        Xn = Xn + rate*g*((cs - X) - (cd - X))
    return np.clip(Xn + r.normal(0, noise, X.shape), 0, 1), R
