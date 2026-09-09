import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'stochastic.py'), encoding='utf-8').read())

def diverging_step(W, Rb, rate=0.1, rs=None):
    """Расхождение вместо схождения: вес УДАЛЯЕТСЯ от среднего по соседям.
    Малые различия усиливаются, а не стираются."""
    r = rs if rs is not None else np.random.default_rng(0)
    deg_in = np.maximum(Rb.sum(axis=1), 1)[:, None]
    deg_out = np.maximum(Rb.sum(axis=0), 1)[None, :]
    inflow = (W.sum(axis=1)/deg_in.ravel())[:, None]
    outflow = (W.sum(axis=0)/deg_out.ravel())[None, :]
    target = 0.5*(inflow + outflow)
    dev = W - target
    out = np.where(Rb, W + rate*dev, 0.0)          # УСИЛИВАЕМ отклонение
    s = out[Rb]
    m = s.mean()
    if m > 0: out = np.where(Rb, out/m, 0.0)       # нормируем, чтобы не улетело
    return out

def split_test(Rb, W0, steps, rate, rs):
    """Расходится ли распределение весов на группы? Меряем бимодальность."""
    W = W0.copy(); hist = []
    for t in range(steps):
        W = diverging_step(W, Rb, rate, rs)
        s = W[Rb]
        # мера расслоения: доля веса в верхней и нижней третях
        q1, q2 = np.quantile(s, [0.33, 0.67])
        hist.append((float(s.std()), float(q2-q1), float(s.max()-s.min())))
    return W, np.array(hist)
