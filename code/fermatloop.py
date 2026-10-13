"""Точная (непрерывная) самая быстрая петля света вокруг дыры через точку на радиусе r_s:
минимум ∫ dt = ∫ √(r'²/A² + r²/A) dφ по замкнутым кривым r(φ), r(0) = r(2π) = r_s."""
import numpy as np, math
from scipy.optimize import minimize
def loop_time(rs, K=120):
    ph = np.linspace(0, 2 * math.pi, K + 1); h = ph[1] - ph[0]
    def T(x):
        r = np.concatenate([[rs], x, [rs]]); A = 1 - 2 / r
        if np.any(A <= 0): return 1e9
        rp = np.diff(r) / h; rm = 0.5 * (r[1:] + r[:-1]); Am = 1 - 2 / rm
        return np.sum(np.sqrt(rp ** 2 / Am ** 2 + rm ** 2 / Am)) * h
    x0 = np.full(K - 1, rs)
    res = minimize(T, x0, method='L-BFGS-B', bounds=[(2.05, 20)] * (K - 1))
    return res.fun
if __name__ == '__main__':
    for rs in (2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 5.0, 6.0):
        print(f'r_s = {rs}: точная петля {loop_time(rs):.2f}M')
