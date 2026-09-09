import numpy as np
from interval_orient import rel_of, interval

def interval_frame2(X, i, j, mid):
    """ДВЕ поперечные оси интервала — полный репер, а не одна ось."""
    if len(mid) < 6: return None
    axis = X[j] - X[i]
    n = np.linalg.norm(axis)
    if n < 1e-12: return None
    e1 = axis/n
    P = X[mid] - 0.5*(X[i]+X[j])
    P = P - np.outer(P @ e1, e1)
    C = P.T @ P
    w, v = np.linalg.eigh(C)
    idx = np.argsort(-w)
    a1, a2 = v[:, idx[0]], v[:, idx[1]]
    return e1, a1, a2, w[idx[0]], w[idx[1]]

def frame_sign(e1, a1, a2):
    """Знак репера: ориентация тройки (e1, a1, a2)."""
    M = np.column_stack([e1[:3], a1[:3], a2[:3]])
    return float(np.sign(np.linalg.det(M)))

def lift_frame(X, Rb, i0, seq):
    """Подъём ПОЛНОГО репера вдоль пути: следим за непрерывностью обеих осей."""
    fr = []
    for j in seq:
        f = interval_frame2(X, i0, j, interval(Rb, i0, j))
        if f is None: return None
        fr.append(f)
    # выравниваем знаки осей по непрерывности
    e1, a1, a2, _, _ = fr[0]
    sgn0 = frame_sign(e1, a1, a2)
    p1, p2 = a1, a2
    flips = 0
    for t in range(1, len(fr)+1):
        e1c, b1, b2, _, _ = fr[t % len(fr)]
        if float(p1 @ b1) < 0: b1 = -b1; flips += 1
        if float(p2 @ b2) < 0: b2 = -b2; flips += 1
        p1, p2 = b1, b2
    sgn1 = frame_sign(fr[0][0], p1, p2)
    return flips, sgn0, sgn1
