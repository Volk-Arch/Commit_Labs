import numpy as np

def recover_scales(R, k, iters=300, rs=None, lr=0.15):
    """Ищем k шкал, пересечение которых даёт порядок R.
    Оптимизация: событие i раньше j <=> X[i] < X[j] по всем шкалам."""
    r = rs if rs is not None else np.random.default_rng(0)
    N = len(R)
    Rb = np.asarray(R, dtype=bool)
    X = r.random((N, k))
    for it in range(iters):
        D = X[None, :, :] - X[:, None, :]     # D[i,j,m] = X[j,m]-X[i,m]
        pred = np.all(D > 0, axis=2)          # i раньше j
        # ошибки: должно быть раньше, но не предсказано
        miss = Rb.T & ~pred                   # Rb[j,i] значит i раньше j
        extra = pred & ~Rb.T
        if not miss.any() and not extra.any(): break
        G = np.zeros_like(X)
        # для пропущенных: тянем X[i] вниз, X[j] вверх по худшей шкале
        mi, mj = np.nonzero(miss)
        if len(mi):
            sub = np.argmin(D[mi, mj], axis=1)
            np.add.at(G, (mi, sub), -1.0)
            np.add.at(G, (mj, sub), +1.0)
        # для лишних: ломаем по самой слабой шкале
        ei, ej = np.nonzero(extra)
        if len(ei):
            sub = np.argmin(D[ei, ej], axis=1)
            np.add.at(G, (ei, sub), +1.0)
            np.add.at(G, (ej, sub), -1.0)
        n = np.maximum(np.abs(G).max(), 1e-9)
        X = np.clip(X + lr*G/n, 0, 1)
    D = X[None, :, :] - X[:, None, :]
    pred = np.all(D > 0, axis=2)
    return X, pred
