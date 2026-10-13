"""Усиление части C: T = 1, много засевов, три быстроты.

Нуль без силы — не вывод. Здесь сила набирается на самом дешёвом случае.
Если при T=1 разницы нет при 60 засевах, то и прослеживать по T нечего.
"""
import numpy as np, math, sys
sys.path.insert(0, '/home/claude/code')
from amplitude import diamond, action_counts, observer_proper

N, T, DENS, TAU, SEEDS = 999, 1.0, 20, 0.22, 60
raps = (0.0, 0.45, 0.9)
acc = {r: [] for r in raps}
B0 = []
for s in range(SEEDS):
    X = diamond(N, T, np.random.default_rng(52000 + 19 * s))
    b0 = action_counts(X)[0]
    B0.append(b0)
    for r in raps:
        Z = observer_proper(DENS, TAU, r, T / 2.0, 0.0)
        acc[r].append(action_counts(np.vstack([X, Z]))[0] - b0)
    del X
print(f'N = {N}, засевов {SEEDS}, наблюдатель {DENS} событий, tau = {TAU}')
print(f'действие пустого мира: {np.mean(B0):.1f} ± '
      f'{np.std(B0, ddof=1)/math.sqrt(SEEDS):.1f} '
      f'(разброс {np.std(B0, ddof=1):.1f})')
print()
print(f"{'быстрота':>9}{'⟨ΔB⟩':>10}{'ст.ош':>9}{'сигм от 0':>11}"
      f"{'промах к ρ=0':>15}{'сигм':>7}")
ref = None
for r in raps:
    v = np.array(acc[r], float)
    se = v.std(ddof=1) / math.sqrt(len(v))
    if ref is None:
        ref = (v.mean(), se); pm, sg = '— отсчёт', ''
    else:
        d = v.mean() - ref[0]; sd = math.hypot(se, ref[1])
        pm, sg = f'{d:+.1f}', f'{abs(d)/max(sd,1e-9):.1f}'
    print(f'{r:>9.2f}{v.mean():>10.1f}{se:>9.1f}'
          f'{abs(v.mean())/max(se,1e-9):>11.1f}{pm:>15}{sg:>7}')
print()
print('ПАРНО по засеву (убирает разброс мира — главный источник шума):')
print(f"{'пара':>14}{'⟨разн⟩':>10}{'ст.ош':>9}{'сигм':>7}{'знак один':>12}")
for a, b in ((0.0, 0.45), (0.0, 0.9), (0.45, 0.9)):
    d = np.array(acc[b], float) - np.array(acc[a], float)
    se = d.std(ddof=1) / math.sqrt(len(d))
    sm = int((d > 0).sum() if d.mean() > 0 else (d < 0).sum())
    print(f'{f"{a} -> {b}":>14}{d.mean():>10.1f}{se:>9.1f}'
          f'{abs(d.mean())/max(se,1e-9):>7.1f}{sm}/{len(d):<7}')
print()
print('  парное сравнение — правильный замер: мир один и тот же,')
print('  меняется только скорость наблюдателя')
