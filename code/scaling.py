"""Доля или число? Что управляет размерностью заряженного мира.

Две серии:
  A. фиксирована ДОЛЯ f=0.05 — если размерность ползёт к 3.04
     с ростом N, то в пределе любая ненулевая доля неотличима от
     половины, и прочитать f из геометрии нельзя;
  B. фиксировано ЧИСЛО античастиц (30) — если размерность одна и та
     же при всех N, управляет число, а не доля.
"""
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from antimatter import rel_charged, closure
from core import dimension

SEEDS = 4
NS = (600, 1200, 2400, 4800)


def measure(N, n_anti, seeds=SEEDS):
    ds = []
    for s in range(seeds):
        rs = np.random.default_rng(7000 + 131 * s)
        X = rs.random((N, 4))
        q = np.ones(N, dtype=int)
        if n_anti > 0:
            q[rs.choice(N, min(n_anti, N), replace=False)] = -1
        C = closure(rel_charged(X, q, 'B'))
        ds.append(dimension(R=C))
    return float(np.mean(ds)), float(np.std(ds))


print('A. ФИКСИРОВАНА ДОЛЯ f = 0.05')
print(f"{'N':>7}{'античастиц':>12}{'размерность':>22}")
for N in NS:
    d, sd = measure(N, int(round(0.05 * N)))
    print(f'{N:>7}{int(round(0.05*N)):>12}{f"{d:.3f} ± {sd:.3f}":>22}')

print()
print('B. ФИКСИРОВАНО ЧИСЛО античастиц = 30')
print(f"{'N':>7}{'доля':>12}{'размерность':>22}")
for N in NS:
    d, sd = measure(N, 30)
    print(f'{N:>7}{30/N:>12.4f}{f"{d:.3f} ± {sd:.3f}":>22}')

print()
print('C. ОДНА античастица на весь мир')
print(f"{'N':>7}{'доля':>12}{'размерность':>22}{'без заряда':>22}")
for N in NS:
    d, sd = measure(N, 1)
    d0, sd0 = measure(N, 0)
    print(f'{N:>7}{1/N:>12.5f}{f"{d:.3f} ± {sd:.3f}":>22}'
          f'{f"{d0:.3f} ± {sd0:.3f}":>22}')
