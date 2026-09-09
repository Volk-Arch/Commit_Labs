import numpy as np

def grow_sequential(N, rule, rs, **kw):
    """Порядок РАСТЁТ: события добавляются по одному,
    каждое выбирает себе прошлое по правилу. Вход не задан заранее."""
    R = np.zeros((N, N), dtype=bool)
    for n in range(1, N):
        past = rule(R, n, rs, **kw)
        if past.any():
            # транзитивное замыкание: наследуем прошлое предков
            idx = np.flatnonzero(past)
            R[:n, n] = past | R[:n, :n][:, idx].any(axis=1)
    return R

def rule_percolation(R, n, rs, p=0.01):
    return rs.random(n) < p

def rule_by_depth(R, n, rs, p=0.01):
    """Предпочитаем недавние события (малая кодглубина)."""
    if n < 2: return rs.random(n) < p
    fut = R[:n, :n].sum(axis=1)
    w = 1.0/(1.0+fut)
    w = w/w.sum()
    m = max(1, int(p*n))
    sel = rs.choice(n, min(m, n), replace=False, p=w)
    out = np.zeros(n, dtype=bool); out[sel] = True
    return out

def rule_by_past(R, n, rs, p=0.01):
    """Предпочитаем события с БОЛЬШИМ прошлым (богатые богатеют)."""
    if n < 2: return rs.random(n) < p
    past = R[:n, :n].sum(axis=1).astype(float)
    w = 1.0+past
    w = w/w.sum()
    m = max(1, int(p*n))
    sel = rs.choice(n, min(m, n), replace=False, p=w)
    out = np.zeros(n, dtype=bool); out[sel] = True
    return out
