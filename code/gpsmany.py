"""Больше спутников при шумных (пуассоновских) тиках: падает ли ошибка как
1/√(число опор)? N=2000, 3 засева. Спутники случайно на сферах r∈[1.6,2.2]."""
import sys, math, numpy as np
sys.path.insert(0, '/home/claude')
from gpsrule import sprinkle, minkowski, compare, stats, V4

def recon(tau, a):
    n, m = tau.shape
    A = np.zeros((n, m - 1, 4)); b = np.zeros((n, m - 1))
    A[:, :, 0] = -2 * (tau[:, 1:] - tau[:, :1]); A[:, :, 1:] = 2 * (a[1:] - a[0])[None]
    b[:] = (np.sum(a[1:] ** 2, 1) - a[0] @ a[0])[None] - tau[:, 1:] ** 2 + tau[:, :1] ** 2
    AT = np.swapaxes(A, 1, 2)
    Y = np.linalg.solve(AT @ A + 1e-12 * np.eye(4), AT @ b[..., None])[..., 0]
    Y[~np.isfinite(Y).all(1) | (np.abs(Y) > 3).any(1)] = 0
    for _ in range(8):
        d = Y[:, None, 1:] - a[None]; rr = np.linalg.norm(d, axis=-1)
        res = Y[:, :1] - tau - rr
        J = np.concatenate([np.ones((n, m, 1)), -d / rr[..., None]], -1); JT = np.swapaxes(J, 1, 2)
        st = np.linalg.solve(JT @ J + 1e-9 * np.eye(4), JT @ res[..., None])[..., 0]
        Y -= np.clip(np.where(np.isfinite(st), st, 0), -.5, .5)
    return Y

N = 2000
print(f"{'опор':>5}{'шаг':>6}{'лишн/R':>8}{'проп/R':>8}{'ош/соб':>8}{'доля':>8}{'разм':>6}{'выс':>6}{'связ':>7}{'спектр':>9}")
for REL in (1.0, 0.5):
    for M in (5, 8, 16, 32):
        acc = []
        for s in range(3):
            rs = np.random.default_rng(900 + s)
            X = sprinkle(N, rs); ell = (V4 / N) ** 0.25; step = REL * ell
            v = rs.normal(size=(M, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
            a = v * rs.uniform(1.6, 2.2, size=(M, 1))
            te = X[:, :1] - np.linalg.norm(X[:, None, 1:] - a[None], axis=-1)
            th = np.empty_like(te)
            for i in range(M):
                ticks = np.cumsum(rs.exponential(step, size=int(14 / step))) - 6
                k = np.searchsorted(ticks, te[:, i]) - 1
                th[:, i] = 0.5 * (ticks[k] + ticks[k + 1])
            Y = recon(th, a)
            R0 = minkowski(X[:, 0], X[:, 1:]); Rg = minkowski(Y[:, 0], Y[:, 1:])
            acc.append(compare(Rg, R0, N) + stats(Rg, True) + stats(R0, True))
        m = np.mean(acc, 0)
        print(f'{M:>5}{REL:>5}ℓ{m[0]:>8.4f}{m[1]:>8.4f}{m[2]:>8.2f}{m[3]:>8.4f}{m[4]:>6.2f}{m[5]:>6.1f}{m[6]:>7.2f}{m[7]:>+9.4f}')
        sys.stdout.flush()
print(f'истина (последний набор): доля {m[8]:.4f} разм {m[9]:.2f} выс {m[10]:.1f} связ {m[11]:.2f} спектр {m[12]:+.4f}')
