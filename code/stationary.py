import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'phasedyn.py'), encoding='utf-8').read())

def evolve(X, steps, rate=0.08, mix=0.5, noise=0.002, rs=None, track=None):
    """Баланс притяжения и отталкивания + шум. Порядок перестраивается,
    величины должны стоять. track — на каких шагах мерить."""
    r = rs if rs is not None else np.random.default_rng(0)
    prev = rel_of(X); rec = []; chs = []
    for t in range(steps):
        R = rel_of(X)
        Rf = R.astype(np.float64)
        npast = np.maximum(Rf.sum(axis=1), 1)[:, None]
        nfut = np.maximum(Rf.sum(axis=0), 1)[:, None]
        cp = (Rf @ X)/npast; cf = (Rf.T @ X)/nfut
        attract = 0.5*(cp+cf); repel = 2*X - attract
        target = mix*attract + (1-mix)*repel
        X = np.clip(X + rate*(target-X) + r.normal(0, noise, X.shape), 0, 1)
        Rn = rel_of(X)
        chs.append(int((Rn ^ prev).sum())); prev = Rn
        if track and t in track:
            rec.append((t, Rn.copy(), X.copy()))
    return X, rec, np.array(chs)
