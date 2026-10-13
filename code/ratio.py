"""Решающее для амплитудного слоя: падает ли sigma(B)/<B> с N.

Интерференция требует sigma(S) ~ hbar; классический предел требует
<S>/hbar >> 1. Оба вместе требуют <S> >> sigma(S). Если отношение
sigma/<> не падает с N, то ни одного hbar, дающего и то и другое, не
существует НИ ПРИ КАКОМ размере мира.
"""
import numpy as np, math, sys
sys.path.insert(0, '/home/claude/code')
from amplitude import diamond, action_counts

print(f"{'N':>7}{'засевов':>9}{'⟨B⟩':>10}{'σ(B)':>10}"
      f"{'σ/⟨⟩':>9}{'⟨B⟩/N':>9}{'σ/√N':>9}")
rows = []
for N, M in ((200, 300), (400, 240), (800, 160), (1600, 90), (3200, 40)):
    B = []
    for s in range(M):
        X = diamond(N, 1.0, np.random.default_rng(60000 + 23 * s))
        B.append(action_counts(X)[0])
        del X
    B = np.array(B, float)
    rows.append((N, B.mean(), B.std(ddof=1)))
    print(f'{N:>7}{M:>9}{B.mean():>10.1f}{B.std(ddof=1):>10.1f}'
          f'{B.std(ddof=1)/abs(B.mean()):>9.2f}{B.mean()/N:>9.3f}'
          f'{B.std(ddof=1)/math.sqrt(N):>9.2f}', flush=True)
A = np.array(rows)
f = lambda y: np.polyfit(np.log(A[:,0]), np.log(np.abs(y)), 1)[0]
print()
print(f'  ⟨B⟩   ~ N^{f(A[:,1]):+.3f}')
print(f'  σ(B)  ~ N^{f(A[:,2]):+.3f}')
print(f'  σ/⟨⟩  ~ N^{f(A[:,2]/A[:,1]):+.3f}')
print()
print('  отношение падает => классический предел появляется при большом N')
print('  растёт или стоит => ни одного годного hbar НИ ПРИ КАКОМ N')
