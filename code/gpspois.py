"""Вариант gpsrule: тики спутников — пуассоновские (случайные события, а не
ровные часы), средний шаг ℓ. Время излучения оценивается серединой между
последним тиком в прошлом и следующим. N=2000, 3 засева."""
import sys, math, numpy as np
sys.path.insert(0, '/home/claude')
from gpsrule import sprinkle, satellites, reconstruct, minkowski, compare, stats, V4
from core import rel_of
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
REL = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
acc = []
for s in range(3):
    rs = np.random.default_rng(500 + s)
    X = sprinkle(N, rs); a = satellites(rs); ell = (V4 / N) ** 0.25; step = REL * ell
    te = X[:, :1] - np.linalg.norm(X[:, None, 1:] - a[None], axis=-1)
    th = np.empty_like(te)
    for i in range(5):
        ticks = np.cumsum(rs.exponential(step, size=int(14 / step)))  - 6
        k = np.searchsorted(ticks, te[:, i]) - 1           # последний тик в прошлом
        th[:, i] = 0.5 * (ticks[k] + ticks[k + 1])
    Y = reconstruct(th, a)
    R0 = minkowski(X[:, 0], X[:, 1:]); Rg = minkowski(Y[:, 0], Y[:, 1:])
    acc.append(compare(Rg, R0, N) + stats(Rg, True) + stats(R0, True))
m = np.mean(acc, 0)
print(f"N={N} пуассоновские тики, шаг {REL}ℓ")
print(f'  лишних/R {m[0]:.4f}  пропущ/R {m[1]:.4f}  ошибок/соб {m[2]:.2f}')
print(f'  GPS:    доля {m[3]:.4f} разм {m[4]:.2f} выс {m[5]:.1f} связ/соб {m[6]:.2f} спектр {m[7]:+.4f}')
print(f'  истина: доля {m[8]:.4f} разм {m[9]:.2f} выс {m[10]:.1f} связ/соб {m[11]:.2f} спектр {m[12]:+.4f}')
