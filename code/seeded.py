import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'conserve.py'), encoding='utf-8').read())

def with_worldline(N, k, chain_len, spread=0.02, rs=None):
    """Сажаем цепочку событий вдоль диагонали (мировая линия) + случайный фон.
    Позиции в [0,1]^k, потом ранжируем — порядок получается честный."""
    r = rs if rs is not None else np.random.default_rng(0)
    t = np.linspace(0.05, 0.95, chain_len)
    line = t[:, None] + r.normal(0, spread, (chain_len, k))
    bg = r.random((N - chain_len, k))
    X = np.clip(np.vstack([line, bg]), 0, 1)
    U = np.array([np.argsort(np.argsort(X[:, m])) for m in range(k)], dtype=np.int64)
    return U, np.arange(chain_len)     # индексы посаженной линии
