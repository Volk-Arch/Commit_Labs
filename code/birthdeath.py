"""Динамика через рождение и гибель событий по квантовому правилу.

Интерференция гасит в одних местах — там события УБИРАЮТСЯ.
В других усиливает — там события РОЖДАЮТСЯ.
Это продолжение того же правила, а не новое.
"""
import numpy as np


def rel_of(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)


def amplitude(X, R, phase=1.0, horizon=4):
    """Амплитуда в каждом событии: сумма по путям из прошлого."""
    N = len(X)
    A = R.astype(np.complex128)*np.exp(1j*phase)
    seed = np.exp(2j*np.pi*X.sum(1))
    acc = seed.copy(); cur = seed.copy()
    for _ in range(horizon):
        cur = A @ cur
        acc = acc + cur
        if np.abs(cur).max() < 1e-14: break
    return acc


def step(X, rate=0.05, phase=1.0, rs=None, spread=0.03):
    """Один шаг: где амплитуда мала — событие гибнет,
    где велика — рождается новое рядом."""
    r = rs if rs is not None else np.random.default_rng(0)
    N, k = X.shape
    R = rel_of(X)
    amp = np.abs(amplitude(X, R, phase))
    q = amp/max(amp.max(), 1e-30)
    n_die = int(rate*N)
    # гибнут события с наименьшей амплитудой
    die = np.argsort(q)[:n_die]
    keep = np.ones(N, bool); keep[die] = False
    Xn = X[keep]
    # рождаются возле событий с наибольшей амплитудой
    live = np.argsort(-q)[:n_die]
    parents = X[live]
    born = np.clip(parents + r.normal(0, spread, parents.shape), 0, 1)
    return np.vstack([Xn, born]), float(q.mean()), int(n_die)
