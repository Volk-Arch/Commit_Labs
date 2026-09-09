import numpy as np

def step(R, n, amp_rule, rs, beta=1.0):
    """Новое событие выбирает прошлое ВЕРОЯТНОСТНО,
    а вероятность считается из УЖЕ СЛУЧИВШЕГОСЯ (амплитуда по путям)."""
    if n == 0: return np.zeros(0, dtype=bool)
    w = amp_rule(R, n)
    if w is None or w.sum() <= 0:
        return rs.random(n) < 0.05
    p = np.abs(w)**2                      # вероятность = |амплитуда|^2
    p = p/p.sum()
    m = max(1, int(0.05*n))
    sel = rs.choice(n, min(m, n), replace=False, p=p)
    out = np.zeros(n, dtype=bool); out[sel] = True
    return out

def amp_paths(R, n):
    """Амплитуда события: сумма по путям из прошлого, с фазой от глубины."""
    if n < 2: return None
    A = R[:n, :n].astype(np.float32)
    past = A.sum(axis=1)
    fut = A.sum(axis=0)
    # фаза от структуры: разность прошлого и будущего
    ph = 2*np.pi*(past-fut)/max(n, 1)
    return np.exp(1j*ph)*(1.0+past)**0.5

def amp_flat(R, n):
    if n < 2: return None
    A = R[:n, :n].astype(np.float32)
    return (1.0+A.sum(axis=1))**0.5

def grow_prob(N, amp_rule, rs):
    R = np.zeros((N, N), dtype=bool)
    for n in range(1, N):
        past = step(R, n, amp_rule, rs)
        if past.any():
            idx = np.flatnonzero(past)
            R[:n, n] = past | R[:n, :n][:, idx].any(axis=1)
    return R
