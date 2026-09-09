import numpy as np
from interval_orient import rel_of, interval, interval_frame
from defects import axis_field

def find_defects(X, F, keys, rad=0.10, thresh=0.4):
    """Ищем места, где поле осей имеет ненулевой индекс."""
    P = np.array([X[j] for j in keys])
    out = []
    for c in range(len(keys)):
        d = np.linalg.norm(P - P[c], axis=1)
        near = np.flatnonzero((d > 1e-9) & (d < rad))
        if len(near) < 6: continue
        ang = np.arctan2(P[near, 2]-P[c, 2], P[near, 1]-P[c, 1])
        order = near[np.argsort(ang)]
        seq = [keys[t] for t in order]
        prev = F[seq[0]]; tot = 0.0
        for t in range(1, len(seq)+1):
            v = F[seq[t % len(seq)]]
            s = 1.0 if float(prev@v) >= 0 else -1.0
            v = s*v
            cr = np.cross(prev[:3], v[:3])
            tot += np.arccos(np.clip(float(prev@v), -1, 1))*np.sign(cr[0] if abs(cr[0]) > 1e-12 else 1)
            prev = v
        w = tot/(2*np.pi)
        if abs(w) > thresh: out.append((keys[c], w))
    return out
