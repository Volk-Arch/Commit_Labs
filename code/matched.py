#!/usr/bin/env python3
"""Доля прямых у дефектов при СОВПАДАЮЩЕЙ ВЫСОТЕ: сигнал или высота.

definfl.py дал: в дефектах доля прямых 0.4078 против 0.3584 (3.0 сигмы,
контроль на псевдодефектах даёт медиану 0.6 и 95-й процентиль 2.0). Но
дефекты сидят НИЖЕ по высоте (4.75 против 5.55, 4.3 сигмы), а у низких
событий связей меньше, значит доля прямых у них выше автоматически.

Весь сигнал может быть высотой. Разделяем: сравнение внутри слоёв
одинаковой высоты, и регрессионный остаток после снятия высоты.
"""
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, '/home/claude/code')
from topo import diamond, cone_rel
from topoinfl import links_of, loop_index

N, T, RHO, NST, MINS, PMIN, SEEDS, SAMP = 2500, 2.0, 0.10, 24, 25, 0.25, 16, 300

rows = []            # (дефект?, доля прямых, высота, число связей)
for t in range(SEEDS):
    rs = np.random.default_rng(5601000 + 977 * t)
    X = diamond(N, T, rs)
    R = cone_rel(X)
    L = links_of(R)
    infl = (L.sum(1) + L.sum(0)) / np.maximum(R.sum(1) + R.sum(0), 1)
    h = np.zeros(len(X), np.int64)
    for i in np.argsort(R.sum(1), kind='stable'):
        i = int(i)
        pre = np.flatnonzero(R[i])
        if len(pre):
            h[i] = h[pre].max() + 1
    i0 = int(np.argmax(R.sum(0) * (X[:, 0] < 0.25 * T)))
    cand = np.flatnonzero(R[:, i0])
    if len(cand) < 300:
        del R, L, X; continue
    for j0 in rs.choice(cand, min(SAMP, len(cand)), replace=False):
        v = loop_index(X, R, i0, cand, int(j0), RHO, NST, MINS, PMIN)
        if v is None:
            continue
        isd = abs(round(v * 2) / 2) >= 0.5
        rows.append((isd, infl[int(j0)], float(h[int(j0)]),
                     float((L[int(j0)].sum() + L[:, int(j0)].sum()))))
    del R, L, X

A = np.array([r[1] for r in rows]); H = np.array([r[2] for r in rows])
D = np.array([r[0] for r in rows], bool)
print(f'всего {len(rows)} индексов, дефектов {int(D.sum())} '
      f'({D.mean():.4f})')
print()
print('СРАВНЕНИЕ ВНУТРИ СЛОЁВ ОДИНАКОВОЙ ВЫСОТЫ')
print(f"{'высота':>8}{'дефектов':>10}{'остальных':>11}{'в дефектах':>12}"
      f"{'в остальных':>13}{'разница':>10}")
num, den = 0.0, 0.0
for hv in sorted(set(H.tolist())):
    m = H == hv
    a, b = A[m & D], A[m & ~D]
    if len(a) < 3 or len(b) < 10:
        continue
    d = a.mean() - b.mean()
    print(f'{hv:>8.0f}{len(a):>10}{len(b):>11}{a.mean():>12.4f}'
          f'{b.mean():>13.4f}{d:>+10.4f}')
    num += d * len(a); den += len(a)
if den > 0:
    print(f'\n  взвешенная разница внутри слоёв: {num/den:+.4f}')
    print(f'  без учёта высоты было:            +0.0494')

# регрессионный остаток
p = np.polyfit(H, A, 2)
res = A - np.polyval(p, H)
d = res[D].mean() - res[~D].mean()
se = math.hypot(res[D].std() / math.sqrt(max(D.sum(), 1)),
                res[~D].std() / math.sqrt(max((~D).sum(), 1)))
print(f'\n  остаток после снятия высоты (квадратичной): '
      f'{d:+.4f} ± {se:.4f}  ({abs(d)/max(se,1e-12):.1f} сигм)')
rs = np.random.default_rng(31337)
zs = []
for _ in range(600):
    pick = rs.choice(len(res), int(D.sum()), replace=False)
    msk = np.zeros(len(res), bool); msk[pick] = True
    dd = res[msk].mean() - res[~msk].mean()
    ss = math.hypot(res[msk].std() / math.sqrt(msk.sum()),
                    res[~msk].std() / math.sqrt((~msk).sum()))
    zs.append(abs(dd) / max(ss, 1e-12))
print(f'  контроль: медиана {np.median(zs):.1f}, '
      f'95-й процентиль {np.percentile(zs,95):.1f}')
print('\n  если остаток держится выше контроля — дефект ВЛИЯЕТ,')
print('  то есть величина сохраняется (топологически) И влияет')
