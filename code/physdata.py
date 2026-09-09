"""Причинные структуры из физических процессов.

Распады частиц образуют настоящий частичный порядок: частица A раньше B,
если B — продукт распада A (прямо или через цепочку).
"""
import numpy as np


def decay_cascade(n_gen, branch, rs, p_stable=0.3):
    """Каскад распадов: частица рождает branch дочерних, часть стабильна.
    Это буквально причинное множество физического процесса."""
    parent = {0: None}
    gen = {0: 0}
    frontier = [0]
    nxt = 1
    for g in range(n_gen):
        new = []
        for p in frontier:
            if rs.random() < p_stable:
                continue
            nb = max(2, int(rs.poisson(branch)))
            for _ in range(nb):
                parent[nxt] = p
                gen[nxt] = g+1
                new.append(nxt)
                nxt += 1
        frontier = new
        if not frontier:
            break
    n = nxt
    R = np.zeros((n, n), dtype=bool)
    for v in range(1, n):
        p = parent[v]
        while p is not None:
            R[v, p] = True
            p = parent[p]
    return R, np.array([gen[i] for i in range(n)])


def scattering_net(n, rs, k=4, spread=0.25):
    """Сеть рассеяний: события связаны, если в световом конусе,
    но с ограничением на импульс (не всё в конусе связано)."""
    X = rs.random((n, k))
    P = rs.normal(0, 1, (n, k-1))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    R = np.zeros((n, n), dtype=bool)
    for i in range(n):
        dt = X[:, 0] - X[i, 0]
        dx = X[:, 1:] - X[i, 1:]
        d = np.linalg.norm(dx, axis=1)
        inside = (dt > 0) & (dt > d)
        # частица летит вдоль своего импульса — связь только вперёд по нему
        with np.errstate(invalid='ignore', divide='ignore'):
            cosang = (dx @ P[i])/np.maximum(d, 1e-12)
        R[inside & (cosang > 1-spread), i] = True
    # транзитивное замыкание
    for _ in range(30):
        Rn = R | ((R.astype(np.float32) @ R.astype(np.float32)) > 0)
        if Rn.sum() == R.sum():
            break
        R = Rn
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)
