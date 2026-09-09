import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'survive.py'), encoding='utf-8').read())

def wave_phases(N, k, mode, rs, nsrc=None):
    """Вместо координат — ФАЗЫ ВОЛН в точке события.
    mode='plane'  — k плоских волн с разными направлениями
    mode='source' — волны от точечных источников (сферические)
    mode='mixed'  — фаза + амплитуда
    """
    pos = rs.random((N, 3))                     # положение события в 3D
    if mode == 'plane':
        K = rs.normal(size=(k, 3))
        K /= np.linalg.norm(K, axis=1, keepdims=True)
        freq = rs.uniform(4, 12, k)
        ph = (pos @ K.T)*freq[None, :]
        return np.mod(ph, 1.0)
    if mode == 'source':
        S = rs.random((k, 3))
        d = np.linalg.norm(pos[:, None, :] - S[None, :, :], axis=2)
        freq = rs.uniform(4, 12, k)
        return np.mod(d*freq[None, :], 1.0)
    if mode == 'coord':                          # контроль: обычные координаты
        return rs.random((N, k))
    return None
