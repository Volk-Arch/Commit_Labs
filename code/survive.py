import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/macro_entropy.py').read())

def mixing_step(X, eps=0.004):
    """Детерминированное перемешивание — то, что даёт стрелу времени."""
    k = X.shape[1]
    d = np.column_stack([X[:, (m+1) % k] for m in range(k)])
    return np.mod(X + eps*np.sin(20*np.pi*d), 1.0)

def make_world(kind, N, k, rs):
    """Разные типы начальных конфигураций."""
    if kind == 'случайный':
        return rs.random((N, k))
    if kind == 'диагональ':
        t = rs.random(N)
        return np.clip(t[:, None] + rs.normal(0, 0.03, (N, k)), 0, 1)
    if kind == 'слоистый':
        X = rs.random((N, k)); X[:, 0] = np.round(X[:, 0]*8)/8
        return X
    if kind == 'кластеры':
        c = rs.random((6, k))
        idx = rs.integers(0, 6, N)
        return np.clip(c[idx] + rs.normal(0, 0.05, (N, k)), 0, 1)
    if kind == 'сжатый':
        return 0.4 + 0.2*rs.random((N, k))
    if kind == 'решётка':
        m = int(np.ceil(N**(1.0/k)))
        g = np.array(np.meshgrid(*[np.linspace(0, 1, m)]*k)).reshape(k, -1).T
        return g[rs.choice(len(g), N, replace=False)] + rs.normal(0, 0.002, (N, k))
    return None

def profile(R):
    """Наблюдаемые: чем характеризуется мир."""
    K = R.T.astype(np.float32)
    try: d = dimension(K)
    except Exception: d = np.nan
    try: o = distinctness(K)
    except Exception: o = np.nan
    N = len(R)
    return np.array([R.sum()/(N*(N-1)/2), d, o, macro_entropy(R)])
