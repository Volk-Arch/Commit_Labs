import numpy as np

def grow_boundary(N, k, rs, width=3, step=0.05):
    """Рост ТОЛЬКО по границе: храним координаты последних `width` слоёв,
    прошлое не хранится вовсе. Память O(width * k), а не O(N^2)."""
    X = np.zeros((N, k))
    depth = np.zeros(N, dtype=np.int64)
    # активный фронт: индексы событий последних width слоёв
    front = [0]
    for n in range(1, N):
        # каждая шкала наследуется от своего предка ИЗ ФРОНТА
        for m in range(k):
            q = front[int(rs.integers(0, len(front)))]
            X[n, m] = X[q, m] + rs.random()*step
        depth[n] = max(depth[q] for q in front) + 1 if front else 0
        front.append(n)
        # держим только последние width слоёв
        top = depth[n]
        front = [v for v in front if depth[v] >= top - width]
        if len(front) > 400:
            front = front[-400:]
    return X, depth

def order_from(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)
