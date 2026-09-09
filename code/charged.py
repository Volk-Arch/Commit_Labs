"""Заряд как ВНЕШНЯЯ метка события.

До сих пор всё выводилось из порядка. Здесь вводится величина,
не сводимая к положению: каждому событию приписан заряд +1 или -1.

Это постулат — и мы меряем, что именно он даёт.
"""
import numpy as np
import sys
import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from deathrules import local_density


def link_phase_charged(R, q):
    """Фаза на связи от зарядов концов: антисимметрична по построению."""
    N = len(R)
    return np.outer(q, q)*0.0 + (q[:, None] - q[None, :])


def field_at(X, q, point, radius=0.3):
    """Поле в точке: сумма зарядов вблизи, взвешенная по расстоянию."""
    d = np.linalg.norm(X - point, axis=1)
    m = d < radius
    if not m.any():
        return 0.0
    w = 1.0/(1.0 + d[m])
    return float((q[m]*w).sum()/w.sum())


def force_on_body(X, q, tag, t, radius=0.5):
    """Сила на тело t: градиент поля, созданного ДРУГИМИ зарядами."""
    mine = tag == t
    if not mine.any():
        return np.zeros(X.shape[1])
    c = X[mine].mean(0)
    qm = float(q[mine].mean())
    other = ~mine
    d = X[other] - c
    r = np.linalg.norm(d, axis=1)
    m = (r < radius) & (r > 1e-9)
    if not m.any():
        return np.zeros(X.shape[1])
    # кулоновская форма: одинаковые заряды ОТТАЛКИВАЮТСЯ,
    # значит сила направлена ОТ заряда того же знака -> минус
    w = q[other][m]/np.maximum(r[m]**2, 1e-6)
    return -qm*(d[m]*w[:, None]).sum(0)/max(m.sum(), 1)
