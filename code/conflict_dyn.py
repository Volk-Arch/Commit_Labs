import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/ensemble.py').read())

def two_views(U, a, b):
    """Два наблюдателя: a видит шкалы a, b видит шкалы b."""
    N = U.shape[1]
    def rel(idx):
        R = np.ones((N, N), dtype=bool)
        for m in idx: R &= (U[m][None, :] < U[m][:, None])
        np.fill_diagonal(R, False); return R
    return rel(a), rel(b)

def reconcile(RA, RB, steps=12):
    """Согласование: там, где наблюдатели противоречат, связь ослабевает у обоих.
    Смотрим, сходится ли процесс и куда."""
    A = RA.astype(np.float64); B = RB.astype(np.float64)
    hist = []
    for t in range(steps):
        conf = np.minimum(A, B.T)                  # A: i<j, B: j<i — противоречие
        A = np.clip(A - 0.5*conf, 0, 1)
        B = np.clip(B - 0.5*conf.T, 0, 1)
        hist.append((float(conf.sum()), float(A.sum()), float(B.sum())))
    return A, B, np.array(hist)
