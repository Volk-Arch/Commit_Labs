import numpy as np
from interval_orient import rel_of, interval, interval_frame

def axis_field(X, Rb, i0, ends):
    """Поле поперечных осей: для каждого конца j — своя ось."""
    out = {}
    for j in ends:
        f = interval_frame(X, i0, j, interval(Rb, i0, j))
        if f is not None: out[j] = f[1]
    return out

def winding_around(X, field, center_js, ref_i, X_all):
    """Индекс особенности: обходим малый круг и считаем поворот оси.
    Для поля НАПРАВЛЕНИЙ (без знака) индекс кратен 1/2."""
    if len(center_js) < 5: return None
    P = X_all[np.array(center_js)]
    c = P.mean(0)
    d = P - c
    ang = np.arctan2(d[:, 2], d[:, 1]) if P.shape[1] > 2 else np.arctan2(d[:, 1], d[:, 0])
    order = np.argsort(ang)
    seq = [center_js[t] for t in order]
    tot = 0.0
    prev = field[seq[0]]
    for t in range(1, len(seq)+1):
        cur = field[seq[t % len(seq)]]
        s = 1.0 if float(prev@cur) >= 0 else -1.0
        cur = s*cur
        c_ = np.clip(float(prev@cur), -1, 1)
        # знак поворота в плоскости обхода
        tot += np.arccos(c_)*np.sign(np.cross(prev[:3], cur[:3])[0] if len(prev) >= 3 else 1)
        prev = cur
    return tot/(2*np.pi)
