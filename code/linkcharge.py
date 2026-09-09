"""Заряд на СВЯЗИ, а не на событии.

Прежде заряд вешался на событие и выводился из его свойств. Здесь он —
независимая величина на канале между двумя событиями, антисимметричная
по направлению: q(a->b) = -q(b->a).
"""
import numpy as np


def links_of(R):
    A = np.asarray(R, dtype=np.float32)
    return np.asarray(R & ((A @ A) == 0), dtype=bool)


def random_link_charge(L, rs, values=(-1.0, 1.0)):
    """Случайный заряд на каждой прямой связи."""
    ii, jj = np.nonzero(L)
    q = rs.choice(values, len(ii))
    Q = np.zeros(L.shape)
    Q[ii, jj] = q
    Q[jj, ii] = -q          # антисимметрия по построению
    return Q


def node_balance(L, Q):
    """Приток минус отток в каждом узле."""
    Lb = np.asarray(L, dtype=bool)
    inflow = (Q*Lb).sum(1)      # по связям, входящим в узел
    outflow = (Q.T*Lb.T).sum(1)
    return inflow - outflow


def divergence(L, Q):
    """Дивергенция: сумма зарядов связей, инцидентных узлу, с учётом знака."""
    Lb = np.asarray(L, dtype=bool)
    # для узла i: связи где i позже (Lb[i,:]) минус где i раньше (Lb[:,i])
    return (Q*Lb).sum(1) - (Q*Lb.T).sum(1)
