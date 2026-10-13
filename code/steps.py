"""Ступенчатость индекса по радиусу: локализован ли заряд.

Если заряд локализован и сохраняется, индекс(R) — ступенчатая функция:
прыгает на +-1/2 при накрытии нового дефекта и стоит между прыжками.
Если бродит плавно — зарядов нет.
"""
import math, numpy as np
from topo import diamond, cone_rel, axis_of, plane_of, loop_index

RADII = (0.05, 0.08, 0.12, 0.17, 0.23)
N, T, S, LOOPS = 3000, 2.0, 3, 160
print(f'N={N}, радиусы {RADII}, засевов {S}')
print('ожидание при локализованном заряде: шаги кратны 1/2, '
      'между ними плато')
print()
seqs = []
for t in range(S):
    rs = np.random.default_rng(221000+977*t)
    X = diamond(N, T, rs); R = cone_rel(X)
    fut = R.sum(0)
    i0 = int(np.argmax(fut*(X[:,0] < 0.25*T)))
    cand = np.flatnonzero(R[:, i0])
    if len(cand) < 300: continue
    rs2 = np.random.default_rng(600+t)
    for j0 in rs2.choice(cand, min(LOOPS, len(cand)), replace=False):
        j0=int(j0)
        try: u,v = plane_of(X, i0, j0)
        except Exception: continue
        row=[]
        for rad in RADII:
            idx = loop_index(X, R, i0, j0, u, v, rad, 24, 25, cand, 0.25)
            row.append(idx)
        if all(x is not None for x in row): seqs.append(row)
    del R, X
A = np.array(seqs)
print(f'полных последовательностей: {len(A)}')
print()
print(f"{'радиус':>8}" + ''.join(f'{r:>10.2f}' for r in RADII))
print(f"{'<индекс>':>8}" + ''.join(f'{x:>+10.4f}' for x in A.mean(0)))
print(f"{'с.к.о.':>8}" + ''.join(f'{x:>10.4f}' for x in A.std(0)))
print(f"{'|инд|>0.25':>8}" + ''.join(f'{x:>10.4f}'
      for x in (np.abs(A)>0.25).mean(0)))
print()
# шаги между соседними радиусами
D = np.diff(A, axis=1).ravel()
D = D[np.isfinite(D)]
print('РАСПРЕДЕЛЕНИЕ ШАГОВ индекса между соседними радиусами')
print(f"{'шаг':>8}{'доля':>10}")
q = np.round(D*4)/4
vals, cnt = np.unique(q, return_counts=True)
for vv, cc in zip(vals, cnt):
    if cc/len(D) > 0.004: print(f'{vv:>+8.2f}{cc/len(D):>10.4f}  ({cc})')
near = np.abs(D - np.round(D*2)/2)
print()
print(f'  шагов всего {len(D)}')
print(f'  доля шагов, попавших в кратное 1/2 с точностью 0.08: '
      f'{(near<0.08).mean():.3f}')
print(f'  доля НУЛЕВЫХ шагов (плато): {(np.abs(D)<0.08).mean():.3f}')
print(f'  медиана |отклонения от кратного 1/2|: {np.median(near):.4f}')
rnd = np.random.default_rng(1).uniform(-0.5,0.5,len(D))
nr = np.abs(rnd - np.round(rnd*2)/2)
print(f'  то же для РАВНОМЕРНОГО шума: в кратное {(nr<0.08).mean():.3f}, '
      f'медиана {np.median(nr):.4f}')
