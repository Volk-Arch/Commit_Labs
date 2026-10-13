import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/charge_dyn.py').read())

def weight_propagate(W, Rb, steps, rate=0.3, damp=0.0):
    """Вес связи зависит от весов СОСЕДНИХ связей (через общее событие).
    Возмущение должно распространяться. Смотрим, с какой скоростью и куда."""
    hist = []
    for t in range(steps):
        deg_in = np.maximum(Rb.sum(axis=1), 1)[:, None]
        deg_out = np.maximum(Rb.sum(axis=0), 1)[None, :]
        inflow = (W.sum(axis=1)/deg_in.ravel())[:, None]
        outflow = (W.sum(axis=0)/deg_out.ravel())[None, :]
        Wn = np.where(Rb, (1-rate)*W + rate*0.5*(inflow+outflow), 0.0)
        if damp: Wn = np.where(Rb, 1.0 + (1-damp)*(Wn-1.0), 0.0)
        W = Wn
        hist.append(W.copy())
    return W, hist

def front_by_depth(W, Rb, D, src_ev, base=1.0, thr=0.01):
    """Где сейчас фронт возмущения: до какой глубины веса заметно отличаются от базы."""
    dev = np.abs(W - base)*Rb
    rows = np.flatnonzero(dev.max(axis=1) > thr)
    cols = np.flatnonzero(dev.max(axis=0) > thr)
    touched = np.union1d(rows, cols)
    if len(touched) == 0: return -1, 0
    return int(D[touched].max() - D[src_ev]), len(touched)
