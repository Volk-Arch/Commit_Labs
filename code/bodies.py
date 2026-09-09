"""Тела в динамике: движутся ли они.

Поле мы мерили статически. Здесь тела эволюционируют по правилу
гибели-рождения вместе с фоном — и видно, сближаются они или расходятся.
"""
import numpy as np
import sys
sys.path.insert(0, '/home/claude')
from deathrules import local_density
from gradrule import density_gradient


def step_grad_bodies(X, tag, rate, rs, radius=0.6, sigma=0.03):
    """Шаг, где новое событие смещается ВДОЛЬ градиента плотности.

    Радиус градиента должен быть больше расстояния между телами,
    иначе они друг друга не видят.
    """
    N, k = X.shape
    n = max(1, int(rate*N))
    g = density_gradient(X, radius)
    dens = local_density(X, 0.15)
    rel = dens/max(dens.mean(), 1e-9)
    w = np.where((rel < 0.7) | (rel > 1.5), 1.0, 0.05)
    die = rs.choice(N, n, replace=False, p=w/w.sum())
    keep = np.ones(N, bool); keep[die] = False
    Xk, tk = X[keep], tag[keep]
    new = np.empty((n, k)); tnew = tag[die].copy()
    for i, (t, pos, gv) in enumerate(zip(tnew, X[die], g[die])):
        if t == 0:
            new[i] = rs.random(k)
        else:
            # смещение ВДОЛЬ градиента: туда, где плотность выше
            new[i] = np.clip(pos + gv + rs.normal(0, sigma, k), 0, 1)
    return np.vstack([Xk, new]), np.concatenate([tk, tnew])


def step_with_bodies(X, tag, rate, rs, lo=0.7, hi=1.5, radius=0.15):
    """Шаг динамики; tag помечает принадлежность телу (0 = фон).

    Новые события рождаются с тем же тегом, что и погибшие —
    иначе тела просто растворятся.
    """
    N, k = X.shape
    n = max(1, int(rate*N))
    dens = local_density(X, radius)
    rel = dens/max(dens.mean(), 1e-9)
    w = np.where((rel < lo) | (rel > hi), 1.0, 0.05)
    die = rs.choice(N, n, replace=False, p=w/w.sum())
    keep = np.ones(N, bool); keep[die] = False
    Xk, tk = X[keep], tag[keep]
    # рождение: событие того же тега появляется рядом с ПОГИБШИМ,
    # а не в центре тела — иначе центр закреплён по построению
    new = np.empty((n, k)); tnew = tag[die].copy()
    for i, (t, pos) in enumerate(zip(tnew, X[die])):
        if t == 0:
            new[i] = rs.random(k)
        else:
            new[i] = np.clip(pos + rs.normal(0, 0.03, k), 0, 1)
    return np.vstack([Xk, new]), np.concatenate([tk, tnew])


def centers(X, tag):
    out = {}
    for t in np.unique(tag):
        if t == 0:
            continue
        out[int(t)] = X[tag == t].mean(0)
    return out
