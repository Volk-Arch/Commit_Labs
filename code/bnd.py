"""Граница или физика: убывает ли зависимость от быстроты при росте мира.

leak.py намерил: добавленных наблюдателем отношений 3184 -> 3490 при
быстроте 0 -> 0.9, то есть +9.6 % при 6.6 сигмы. Если это граница (ромб
не инвариантен к бусту), зависимость обязана УБЫВАТЬ при росте мира при
ТОМ ЖЕ наблюдателе. Если физика — останется.

Наблюдатель один и тот же во всех мирах: та же мощность, то же
собственное время, то же место.
"""
import numpy as np, math, sys
sys.path.insert(0, '/home/claude/code')
from ens2 import action_and_pairs
from particles import diamond, rel_of
from leak import obs_proper, fits

TAU, DENS, SEEDS = 0.22, 20, 18
print(f'наблюдатель: {DENS} событий, собственное время {TAU} — '
      f'один и тот же во всех мирах')
print(f'засевов {SEEDS}')
print()
print(f"{'N':>7}{'ell':>8}{'tau/ell':>9}{'Δотн ρ=0':>11}{'Δотн ρ=0.9':>12}"
      f"{'рост %':>9}{'сигм':>7}{'Δотн/N':>10}")
rows = []
for N in (500, 1000, 2000, 4000):
    ell = (math.pi/24.0/N)**0.25
    if not fits(obs_proper(DENS, TAU, 0.9, 0.5, 0.0)):
        print(f'{N:>7}  наблюдатель вылезает'); continue
    d0, d9 = [], []
    for s in range(SEEDS):
        X = diamond(N, np.random.default_rng(900+31*s))
        P0 = int(rel_of(X).sum())
        for rap, acc in ((0.0, d0), (0.9, d9)):
            Z = obs_proper(DENS, TAU, rap, 0.5, 0.0)
            acc.append(int(rel_of(np.vstack([X, Z])).sum()) - P0)
        del X
    d0 = np.array(d0, float); d9 = np.array(d9, float)
    dd = d9.mean() - d0.mean()
    sd = math.hypot(d0.std(ddof=1)/math.sqrt(len(d0)),
                    d9.std(ddof=1)/math.sqrt(len(d9)))
    rows.append((N, 100*dd/d0.mean(), abs(dd)/max(sd,1e-9)))
    print(f'{N:>7}{ell:>8.4f}{TAU/ell:>9.2f}{d0.mean():>11.0f}'
          f'{d9.mean():>12.0f}{100*dd/d0.mean():>9.2f}'
          f'{abs(dd)/max(sd,1e-9):>7.1f}{d0.mean()/N:>10.3f}')
A = np.array(rows)
if len(A) >= 3:
    b = np.polyfit(np.log(A[:,0]), np.log(np.abs(A[:,1])), 1)[0]
    print()
    print(f'  рост в % ~ N^{b:+.3f}')
    print('  убывает => ГРАНИЦА; стоит => физика (протечка)')
