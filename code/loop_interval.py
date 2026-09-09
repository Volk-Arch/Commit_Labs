import numpy as np
from interval_orient import rel_of, interval, interval_frame

def build_loop(X, Rb, i0, j0, nsteps=24, radius=0.10, minsize=25):
    """Замкнутый обход: конец j движется по кругу вокруг j0 в поперечной плоскости.
    Возвращает последовательность концов, образующих петлю."""
    k = X.shape[1]
    axis = X[j0] - X[i0]; axis /= np.linalg.norm(axis)
    # два поперечных направления
    a = np.eye(k)[np.argmin(np.abs(axis))]
    e1 = a - (a@axis)*axis; e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis[:3], e1[:3]) if k >= 3 else None
    if e2 is None or np.linalg.norm(e2) < 1e-12: return None
    e2 = np.concatenate([e2, np.zeros(k-3)]) if k > 3 else e2
    e2 /= np.linalg.norm(e2)
    seq = []
    for t in range(nsteps):
        th = 2*np.pi*t/nsteps
        target = X[j0] + radius*(np.cos(th)*e1 + np.sin(th)*e2)
        # ближайший реальный конец к цели
        cand = [j for j in range(len(X)) if Rb[i0, j]]
        if not cand: return None
        C = np.array(cand)
        d = np.linalg.norm(X[C]-target, axis=1)
        for idx in np.argsort(d)[:12]:
            j = int(C[idx])
            if len(interval(Rb, i0, j)) >= minsize:
                seq.append(j); break
    return seq if len(seq) == nsteps else None

def lift_axis(X, Rb, i0, seq):
    """Подъём поперечной оси вдоль петли: следим за непрерывностью знака.
    Возвращает суммарный поворот в единицах пи и знак после обхода."""
    axes = []
    for j in seq:
        f = interval_frame(X, i0, j, interval(Rb, i0, j))
        if f is None: return None
        axes.append(f[1])
    tot = 0.0; sign = 1.0
    prev = axes[0]
    for t in range(1, len(axes)+1):
        cur = axes[t % len(axes)]
        d = float(prev @ cur)
        if d < 0:                      # ось — направление без знака, выравниваем
            cur = -cur; sign = -sign
            d = -d
        tot += np.arccos(np.clip(d, -1, 1))
        prev = cur
    return tot/np.pi, sign
