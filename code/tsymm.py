import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/survey.py').read())

def reverse_time(U):
    """Обращение времени: все ранги переворачиваются. Прошлое становится будущим."""
    N = U.shape[1]
    return (N - 1) - U

def pair_symmetry_features(U, R, want=6000, rs=None):
    """Величины пары и их поведение при трёх преобразованиях:
    P — обмен двух шкал (зеркало), T — обращение времени, PT — обе."""
    r = rs if rs is not None else np.random.default_rng(0)
    k, N = U.shape
    Rb = np.ascontiguousarray(R, dtype=bool)
    Rk = U.T.astype(float)/N
    rows = []; tries = 0
    while len(rows) < want and tries < want*40:
        tries += 1
        i, j = (int(v) for v in r.integers(0, N, 2))
        if i == j: continue
        if Rb[i, j]: i, j = j, i
        dr = Rk[j] - Rk[i]
        a, b, c = dr[0], dr[1], dr[2]
        par = np.sign(a-b)*np.sign(b-c)*np.sign(a-c)     # P-нечётна
        tot = a + b + c                                   # T-нечётна (все меняют знак)
        # T-нечётная и P-нечётная одновременно
        pt = par*np.sign(tot)
        # ещё кандидат: знак произведения
        prod = np.sign(a*b*c)
        rows.append([float(par), float(np.sign(tot)), float(pt), float(prod),
                     float(np.abs(dr).sum()), float(Rb[j, i])])
    names = ['P: чётность прир.', 'T: знак суммы', 'PT: произведение',
             'знак произв. прир.', 'длина прир.', 'связаны']
    return np.array(rows, dtype=float), names
