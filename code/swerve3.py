"""Контроль сноса быстроты: то же правило K=8 на Пуассоне и на кристалле, 150 шагов."""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
from swerve import Poisson, Crystal, mdot, u_of
from swerve2 import walk2
t0 = time.time(); rng = np.random.default_rng(9)
for name, mk in (('кристалл', Crystal), ('Пуассон', Poisson)):
    for eta0 in (0.5, 2.5):
        fin = []
        for p in range(14):
            U = walk2(mk(int(rng.integers(1e9))), u_of(eta0, rng.normal(size=3)), 150, 8); fin.append(math.acosh(U[-1, 0]))
        print(f'{name:9s} η₀={eta0}: быстрота через 150 шагов {np.mean(fin):.2f} ± {np.std(fin)/math.sqrt(len(fin)):.2f}   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
