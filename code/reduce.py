import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'infocount.py'), encoding='utf-8').read())

def from_two(N, k, rs):
    """Кладём ДВА числа — сумму и разброс — и восстанавливаем k шкал.
    Проверяем, даст ли это тот же мир."""
    S = rs.random(N)                     # «время»
    V = rs.random(N)                     # «разброс»
    # реконструируем k шкал так, чтобы у них были заданные среднее и разброс
    base = rs.normal(size=(N, k))
    base -= base.mean(1, keepdims=True)
    base /= np.maximum(base.std(1, keepdims=True), 1e-9)
    X = S[:, None] + V[:, None]*base
    return np.mod(X, 1.0), S, V

def from_two_fixed(N, k, rs):
    """То же, но 'форма' одна и та же для всех событий — только сумма и разброс свои."""
    shape = rs.normal(size=k); shape -= shape.mean(); shape /= shape.std()
    S = rs.random(N); V = rs.random(N)
    X = S[:, None] + V[:, None]*shape[None, :]
    return np.mod(X, 1.0), S, V
