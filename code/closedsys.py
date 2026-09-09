"""Замкнутая система: динамика без притока извне.

Для первого начала нужен баланс, а он возможен только в замкнутой системе.
У нас события рождались свободно — то есть из ниоткуда. Здесь новые события
берутся из ФИКСИРОВАННОГО запаса: сколько погибло, столько и родилось,
и координаты берутся из тех же, что освободились.
"""
import numpy as np


def rel_of(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k):
        R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)


def links_of(R):
    A = R.astype(np.float32)
    return R & ((A @ A) == 0)


def frac_direct(X):
    R = rel_of(X)
    L = links_of(R)
    v = float(L.sum()/max(R.sum(), 1))
    del R, L
    return v


def step_closed(X, rate, rs, mode="shuffle"):
    """Замкнутый шаг: координаты НЕ создаются и НЕ уничтожаются.

    'shuffle'  — координаты погибших перераспределяются между шкалами
    'swap'     — два события меняются частью координат
    'diffuse'  — координата переходит от одного события к другому
    """
    N, k = X.shape
    n = max(1, int(rate*N))
    Xn = X.copy()
    if mode == "shuffle":
        idx = rs.choice(N, n, replace=False)
        block = Xn[idx].copy()
        # перемешиваем координаты внутри блока по каждой шкале отдельно
        for m in range(k):
            block[:, m] = rs.permutation(block[:, m])
        Xn[idx] = block
    elif mode == "swap":
        # ⚠ a и b должны быть НЕПЕРЕСЕКАЮЩИМИСЯ: если событие попадёт в оба
        # списка, обмен потеряет координату и сумма по шкале не сохранится
        pick = rs.choice(N, 2*n, replace=False)
        a, b = pick[:n], pick[n:]
        m = rs.integers(0, k, n)
        tmp = Xn[a, m].copy()
        Xn[a, m] = Xn[b, m]
        Xn[b, m] = tmp
    elif mode == "diffuse":
        pick = rs.choice(N, 2*n, replace=False)
        a, b = pick[:n], pick[n:]
        m = rs.integers(0, k, n)
        d = 0.5*(Xn[b, m] - Xn[a, m])
        Xn[a, m] += d
        Xn[b, m] -= d
    return Xn
