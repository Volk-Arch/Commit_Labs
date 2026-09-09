import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/weightdyn.py').read())

def count_in_interval(X, i, j):
    """Сколько событий в интервале Александрова между i и j."""
    k = X.shape[1]
    after_i = np.all(X > X[i], axis=1)
    before_j = np.all(X < X[j], axis=1)
    return int((after_i & before_j).sum())

def volume_of(X, i, j):
    """Объём интервала в фазовых координатах — произведение приращений."""
    return float(np.prod(np.abs(X[j] - X[i])))
