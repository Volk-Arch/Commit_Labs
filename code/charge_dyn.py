import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/charge_conserve.py').read())

def full_margin(X):
    med = np.median(X, axis=1, keepdims=True); dev = np.abs(X - med)
    srt = np.sort(dev, axis=1)
    return float((srt[:, -1] - srt[:, -2]).sum())

def step_with_charge(X, g=0.0, eps=0.004, mode='chaos', rs=None, noise=0.0):
    """Детерминированное перемешивание ПЛЮС взаимодействие по заряду.
    Заряд действует на ФАЗЫ через притяжение своих (проверяем, что он делает с сохранением)."""
    r = rs if rs is not None else np.random.default_rng(0)
    k = X.shape[1]
    if mode == 'chaos':
        d = np.column_stack([X[:, (m+1) % k] for m in range(k)])
        Xn = np.mod(X + eps*np.sin(20*np.pi*d), 1.0)
    else:
        Xn = np.mod(X + eps*np.array([1.0, np.sqrt(2)-1, np.sqrt(3)-1])[None, :], 1.0)
    if g != 0.0:
        R = rel_of(X); Rf = R.astype(np.float64)
        lab, _ = labels_of(X)
        same = (lab[None, :] == lab[:, None]) & (lab[None, :] >= 0) & (lab[:, None] >= 0)
        S = (Rf > 0) & same
        ns = np.maximum(S.sum(axis=1), 1)[:, None]
        cs = (S.astype(np.float64) @ X)/ns
        Xn = np.mod(Xn + eps*g*(cs - X), 1.0)
    if noise: Xn = np.mod(Xn + r.normal(0, noise, X.shape), 1.0)
    return Xn
