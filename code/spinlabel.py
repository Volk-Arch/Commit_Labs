import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'chiral2.py'), encoding='utf-8').read())
rng15 = np.random.default_rng(263)

def rel_spin(U, s, mode, mask=None):
    """У события есть бинарная метка s (±1) — аналог спиральности.
    mode='none'   — метка не участвует (контроль)
    mode='pair'   — киральное условие применяется к парам, где обе метки +1
    mode='full'   — внутри таких пар отсечение ПОЛНОЕ: связь только при чётной ранжировке
                    И при согласии метки с ориентацией приращений
    """
    N = U.shape[1]
    R = base_rel(U)
    if mode == 'none': return R
    a, b, c = [U[m][None, :] - U[m][:, None] for m in range(3)]
    par = np.sign(a-b)*np.sign(b-c)*np.sign(a-c)
    both = (s[None, :] > 0) & (s[:, None] > 0)
    if mask is not None: both = both & mask
    if mode == 'pair':
        return R & (~both | (par > 0))
    # full: метка задаёт, какая чётность разрешена — но метка сама по себе
    # к вращению шкал безразлична, значит вращательная симметрия должна выжить
    want = np.where(s[None, :] > 0, 1, -1)
    return R & (~both | (par == want))
