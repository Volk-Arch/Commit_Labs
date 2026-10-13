import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/global_topo.py').read())

def iterate_weights(R, W0=None, steps=30, mode='avg', rate=0.5):
    """Вес связи пересчитывается по весам СОСЕДНИХ связей и кладётся обратно.
    mode='avg'  — новый вес = среднее весов входящих и исходящих связей
    mode='fwd'  — только по ВХОДЯЩИМ (прошлое влияет на будущее)
    mode='bwd'  — только по ИСХОДЯЩИМ (будущее влияет на прошлое)"""
    Rb = np.ascontiguousarray(R, dtype=bool)
    W = (np.ones_like(Rb, dtype=np.float64) if W0 is None else W0.copy())*Rb
    hist = []
    deg_in = np.maximum(Rb.sum(axis=1), 1)[:, None]
    deg_out = np.maximum(Rb.sum(axis=0), 1)[None, :]
    for t in range(steps):
        inflow = (W.sum(axis=1)/deg_in.ravel())[:, None]      # средний вес связей, входящих в j
        outflow = (W.sum(axis=0)/deg_out.ravel())[None, :]    # исходящих из i
        if mode == 'avg':   target = 0.5*(inflow + outflow)
        elif mode == 'fwd': target = np.broadcast_to(outflow, W.shape)
        else:               target = np.broadcast_to(inflow, W.shape)
        W = np.where(Rb, (1-rate)*W + rate*target, 0.0)
        s = W[Rb]
        hist.append((float(s.mean()), float(s.std())))
    return W, np.array(hist)
