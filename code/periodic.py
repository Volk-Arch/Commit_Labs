"""Мир без границ: периодическое замыкание.

В кубе градиент плотности направлен внутрь (у краёв соседей меньше),
и тела сносит к центру. При замыкании краёв нет — снос исчезает.
"""
import numpy as np


def wrap_diff(A, B):
    """Разность с учётом замыкания: кратчайший путь по тору."""
    d = A - B
    return d - np.round(d)


def periodic_density(X, radius, chunk=256):
    N, k = X.shape
    out = np.zeros(N)
    r2 = radius*radius
    for a in range(0, N, chunk):
        b = min(a+chunk, N)
        d = X[a:b, None, :] - X[None, :, :]
        d -= np.round(d)
        d2 = (d*d).sum(-1)
        out[a:b] = np.count_nonzero(d2 < r2, axis=1) - 1
        del d, d2
    return out


def periodic_gradient(X, radius, chunk=256):
    N, k = X.shape
    g = np.zeros((N, k))
    r2 = radius*radius
    for a in range(0, N, chunk):
        b = min(a+chunk, N)
        d = X[None, :, :] - X[a:b, None, :]
        d -= np.round(d)
        d2 = (d*d).sum(-1)
        near = d2 < r2
        for i in range(b-a):
            m = near[i].copy()
            m[a+i] = False
            if m.sum():
                g[a+i] = d[i][m].mean(0)
        del d, d2, near
    return g


def periodic_center(P):
    """Центр масс на торе: через средний угол."""
    ang = 2*np.pi*P
    c = np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0))/(2*np.pi)
    return np.mod(c, 1.0)


def periodic_dist(a, b):
    d = a - b
    d -= np.round(d)
    return float(np.linalg.norm(d))


def step_periodic(X, tag, rate, rs, radius=0.35, sigma=0.03, gain=1.0):
    """Шаг на торе: гибель по окну плотности, рождение вдоль градиента."""
    N, k = X.shape
    n = max(1, int(rate*N))
    dens = periodic_density(X, 0.15)
    rel = dens/max(dens.mean(), 1e-9)
    w = np.where((rel < 0.7) | (rel > 1.5), 1.0, 0.05)
    die = rs.choice(N, n, replace=False, p=w/w.sum())
    g = periodic_gradient(X, radius)
    keep = np.ones(N, bool); keep[die] = False
    Xk, tk = X[keep], tag[keep]
    new = np.empty((n, k)); tnew = tag[die].copy()
    for i, (t, pos, gv) in enumerate(zip(tnew, X[die], g[die])):
        if t == 0:
            new[i] = rs.random(k)
        else:
            new[i] = np.mod(pos + gain*gv + rs.normal(0, sigma, k), 1.0)
    return np.vstack([Xk, new]), np.concatenate([tk, tnew])
