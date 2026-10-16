"""Дополнение: (а) углы, у которых тень = мир (рациональные повороты, они есть в Q(τ)), состояние —
золотое (тень состояния отличается); (б) общий случай: углы мира и тени независимы (Q(τ) плотно
в R², так что допустима любая пара). Максимум CHSH при P ≥ 0 и без сигнала."""
import math, numpy as np
from scipy.optimize import minimize
from goldbell import evaluate, TAU
rs = np.random.default_rng(1)
def run(sw, ss, free, tries=300):
    best = (0, None)
    for _ in range(tries):
        x0 = rs.uniform(0, 2 * math.pi, 8 if free else 4)
        def unpack(v):
            if free: ax, bx, axs, bxs = v[:2], v[2:4], v[4:6], v[6:8]
            else: ax, bx = v[:2], v[2:4]; axs, bxs = ax, bx
            return ax, bx, axs, bxs
        def f(v):
            ax, bx, axs, bxs = unpack(v)
            m = {ax[0]: axs[0], ax[1]: axs[1], bx[0]: bxs[0], bx[1]: bxs[1]}
            r = evaluate(sw, ss, ax, bx, lambda th: m[th])
            if r is None: return 5.0
            S, ns, _ = r
            return -abs(S) + 50 * ns
        res = minimize(f, x0, method='Nelder-Mead', options=dict(maxiter=3000, xatol=1e-9, fatol=1e-12))
        ax, bx, axs, bxs = unpack(res.x); m = {ax[0]: axs[0], ax[1]: axs[1], bx[0]: bxs[0], bx[1]: bxs[1]}
        r = evaluate(sw, ss, ax, bx, lambda th: m[th])
        if r and r[1] < 1e-7 and abs(r[0]) > best[0]: best = (abs(r[0]), np.round(np.degrees(res.x) % 360, 1))
    return best
print('(а) углы с тенью = миру; состояние s = (1, x): мир x, тень x′')
for (a, b) in ((1, 0), (0, 1), (1, 1), (-1, 1), (2, -1), (1, -1)):
    x = a + b * TAU; xs = a - b / TAU
    print(f'   x = {a}+{b}τ = {x:+.3f}, тень {xs:+.3f}:  max |S| = {run((1, x), (1, xs), False)[0]:.4f}')
print('(б) углы мира и тени независимы')
for (a, b) in ((1, 0), (0, 1), (1, 1), (-1, 1)):
    x = a + b * TAU; xs = a - b / TAU
    r = run((1, x), (1, xs), True, 200)
    print(f'   x = {a}+{b}τ:  max |S| = {r[0]:.4f}  углы {r[1]}')
