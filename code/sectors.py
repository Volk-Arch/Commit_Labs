import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
from itertools import combinations
exec(open('/home/claude/intrinsic.py').read())

def sector(U, idx):
    """Причинная структура, видимая через подмножество шкал idx."""
    N = U.shape[1]
    R = np.ones((N, N), dtype=bool)
    for m in idx: R &= (U[m][None, :] < U[m][:, None])
    np.fill_diagonal(R, False)
    return R

def event_profile(U, sets):
    """Для каждого события: сколько связей оно имеет в каждом секторе."""
    prof = []
    for s in sets:
        R = sector(U, s)
        prof.append(R.sum(axis=0) + R.sum(axis=1))
    return np.array(prof)          # [сектор, событие]
