import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'weightdyn.py'), encoding='utf-8').read())

def count_in_interval(X, i, j):
    """Сколько событий в интервале Александрова между i и j."""
    k = X.shape[1]
    after_i = np.all(X > X[i], axis=1)
    before_j = np.all(X < X[j], axis=1)
    return int((after_i & before_j).sum())

def volume_of(X, i, j):
    """Объём интервала в фазовых координатах — произведение приращений."""
    return float(np.prod(np.abs(X[j] - X[i])))
