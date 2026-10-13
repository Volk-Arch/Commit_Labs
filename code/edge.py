"""Остаток взвешенной суммы: граничный он или объёмный."""
import numpy as np
from weight import diamond, cone_rel, between, per_event

print('остаток по глубине события в мире, eps = 0.2 и 0.5')
print('глубина = min(t, T-t) - |x|, то есть собственное время до края ромба')
print()
print(f"{'T':>5}{'N':>7}{'eps':>6}{'весь мир':>22}{'внутр. половина':>22}{'внешняя':>22}")
for T in (1.5, 2.0, 2.5):
    rs = np.random.default_rng(900)
    X = diamond(T, rs)
    R = cone_rel(X); btw = between(R)
    r = np.linalg.norm(X[:, 1:], axis=1)
    depth = np.minimum(X[:, 0], T - X[:, 0]) - r
    med = np.median(depth)
    deep, shal = depth >= med, depth < med
    for eps in (0.5, 0.2):
        s, _ = per_event(R, btw, eps)
        f = lambda m: (s[m].mean(), s[m].std()/np.sqrt(m.sum()))
        a, b, c = f(np.ones(len(s), bool)), f(deep), f(shal)
        print(f'{T:>5.1f}{len(X):>7}{eps:>6.2f}'
              f'{a[0]:>14.4f} ±{a[1]:<7.4f}{b[0]:>14.4f} ±{b[1]:<7.4f}'
              f'{c[0]:>14.4f} ±{c[1]:<7.4f}')
    del R, btw, X
