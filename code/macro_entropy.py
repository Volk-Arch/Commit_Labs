import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/stationary.py').read())

def abundances(R, kmax=6):
    """Обилия интервалов: сколько пар с ровно k элементами между. Это МАКРО-описание."""
    C = np.asarray(R, dtype=np.float32).T
    btw = (C @ C).astype(np.int32)
    b = btw[C > 0]
    h = np.array([int((b == k).sum()) for k in range(kmax+1)] + [int((b > kmax).sum())],
                 dtype=np.float64)
    return h

def macro_entropy(R, kmax=6):
    """Энтропия распределения обилий: насколько макросостояние «размазано».
    Больше энтропия = больше микроконфигураций дают такую картину."""
    h = abundances(R, kmax)
    p = h/max(h.sum(), 1)
    p = p[p > 0]
    return float(-(p*np.log(p)).sum())

def past_diversity(R, samples=3000, rs=None):
    """Эффективное число различных прошлых — прямой аналог числа микросостояний."""
    r = rs if rs is not None else np.random.default_rng(0)
    Rb = np.ascontiguousarray(R, dtype=bool); N = len(Rb)
    idx = r.choice(N, min(samples, N), replace=False)
    packed = np.packbits(Rb[idx], axis=1)
    _, cnt = np.unique(packed, axis=0, return_counts=True)
    p = cnt/cnt.sum()
    return float(np.exp(-(p*np.log(p)).sum()))
