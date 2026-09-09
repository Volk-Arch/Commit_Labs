import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/conflict_dyn.py').read())

def noisy_step(W, Rb, rate=0.15, rs=None):
    """Случайный шаг: вес связи слегка смещается к среднему по соседям ПЛЮС шум.
    Усреднение теряет информацию, шум не даёт застыть."""
    r = rs if rs is not None else np.random.default_rng(0)
    deg_in = np.maximum(Rb.sum(axis=1), 1)[:, None]
    deg_out = np.maximum(Rb.sum(axis=0), 1)[None, :]
    inflow = (W.sum(axis=1)/deg_in.ravel())[:, None]
    outflow = (W.sum(axis=0)/deg_out.ravel())[None, :]
    target = 0.5*(inflow + outflow)
    noise = r.normal(0, rate*0.3, W.shape)
    return np.where(Rb, np.clip((1-rate)*W + rate*target + noise*rate, 0, 3), 0.0)

def converge_test(Rb, starts, steps=40, rate=0.15, seed=0):
    """Сходятся ли РАЗНЫЕ начальные состояния к одному конечному?
    Это и есть потеря информации — признак необратимости."""
    r = np.random.default_rng(seed)
    finals = []
    for W0 in starts:
        W = W0.copy()
        for _ in range(steps):
            W = noisy_step(W, Rb, rate, r)
        finals.append(W[Rb])
    F = np.array(finals)
    # попарное расхождение конечных состояний
    d = []
    for a in range(len(F)):
        for b in range(a+1, len(F)):
            d.append(float(np.abs(F[a]-F[b]).mean()))
    d0 = []
    S = np.array([w[Rb] for w in starts])
    for a in range(len(S)):
        for b in range(a+1, len(S)):
            d0.append(float(np.abs(S[a]-S[b]).mean()))
    return float(np.mean(d0)), float(np.mean(d)), F
