"""Сохранение заряда: ток и непрерывность.

Заряд сохраняется, если новое событие наследует заряд погибшего.
Вопрос: есть ли ток, и выполняется ли уравнение непрерывности —
изменение заряда в области равно потоку через её границу.
"""
import numpy as np
import sys
import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from deathrules import local_density


def step_conserve(X, q, rate, rs, radius=0.15, mode="inherit"):
    """Шаг с сохранением заряда.

    inherit — новое событие наследует заряд погибшего (заряд сохраняется)
    free    — новое событие нейтрально (заряд утекает)
    """
    N, k = X.shape
    n = max(1, int(rate*N))
    d = local_density(X, radius)
    rel = d/max(d.mean(), 1e-9)
    w = np.where((rel < 0.7) | (rel > 1.5), 1.0, 0.05)
    die = rs.choice(N, n, replace=False, p=w/w.sum())
    keep = np.ones(N, bool); keep[die] = False
    newX = rs.random((n, k))
    newq = q[die] if mode == "inherit" else np.zeros(n)
    return np.vstack([X[keep], newX]), np.concatenate([q[keep], newq]), X[die], newX, q[die]


def charge_in(X, q, centre, radius):
    d = np.linalg.norm(X - centre, axis=1)
    return float(q[d < radius].sum())


def flux_through(old_pos, new_pos, qmoved, centre, radius):
    """Поток заряда через границу: сколько заряда пересекло её за шаг."""
    din = np.linalg.norm(old_pos - centre, axis=1) < radius
    dout = np.linalg.norm(new_pos - centre, axis=1) < radius
    # заряд вышел: был внутри, стал снаружи
    out = float(qmoved[din & ~dout].sum())
    inn = float(qmoved[~din & dout].sum())
    return inn - out
