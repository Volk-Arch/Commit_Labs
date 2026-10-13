"""Универсален ли коэффициент задержки?

Прирост собственного времени почти не зависит от размера интервала и
пропорционален высоте массы. Отношение прирост/высота — безразмерное
число. Если оно одинаково при разных k, это величина, которую мы не
задавали.

Мерим в режиме слабого поля: масса много меньше населения интервала.
Контроль рядом — те же события, рассыпанные по миру.
"""
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from chaingrav import build, measure, sample_pairs

N, M, R_BALL, MINVOL = 2000, 100, 0.12, 150


def coeff(k, seed):
    Xb, Xbody, Xunif = build(N, M, k, R_BALL, seed)
    rs = np.random.default_rng(seed + 1)
    pairs = sample_pairs(Xb, N, 350, MINVOL, rs)
    if len(pairs) < 40:
        return None
    base = measure(Xb, N, pairs)
    out = {}
    for name, X in (('тело', Xbody), ('контроль', Xunif)):
        A = measure(X, N, pairs)
        msk = A[:, 3] >= 3                      # выше порога
        if msk.sum() < 10:
            out[name] = None
            continue
        dt = (A[msk, 0] - base[msk, 0]).astype(float)
        ha = A[msk, 3].astype(float)
        ratio = dt / ha
        out[name] = (ratio.mean(), ratio.std() / np.sqrt(len(ratio)),
                     int(msk.sum()), float(base[:, 1].mean()))
    return out


print(f'N={N}, масса M={M}, интервалы V_bg >= {MINVOL}, потолок >= 3')
print()
print(f"{'k':>3}{'M/V_bg':>9}{'тело: прирост/высота':>24}{'пар':>6}"
      f"{'контроль':>22}{'пар':>6}")
vals = []
for k in (3, 4, 5, 6):
    agg = {'тело': [], 'контроль': []}
    meta = None
    for s in (5, 23, 41):
        r = coeff(k, s)
        if r is None:
            continue
        for nm in agg:
            if r[nm]:
                agg[nm].append(r[nm][0])
                if nm == 'тело':
                    meta = (r[nm][2], r[nm][3])
    if not agg['тело']:
        print(f'{k:>3}{"мало пар":>9}')
        continue
    bm, bs = float(np.mean(agg['тело'])), float(np.std(agg['тело']))
    if agg['контроль']:
        cm, cs = float(np.mean(agg['контроль'])), float(np.std(agg['контроль']))
        ctxt = f'{cm:.3f} ± {cs:.3f}'
    else:
        cm, ctxt = float('nan'), 'нет пар'
    vals.append((k, bm, bs))
    print(f'{k:>3}{M/meta[1]:>9.3f}{f"{bm:.3f} ± {bs:.3f}":>24}{meta[0]:>6}'
          f'{ctxt:>22}{len(agg["контроль"]):>6}')

print()
if len(vals) >= 3:
    ms = np.array([v[1] for v in vals])
    ss = np.array([max(v[2], 1e-6) for v in vals])
    w = 1 / ss ** 2
    mu = float((ms * w).sum() / w.sum())
    chi2 = float((((ms - mu) / ss) ** 2).sum())
    print(f'  общее среднее: {mu:.3f}')
    print(f'  хи-квадрат на совместимость: {chi2:.1f} при {len(vals)-1} ст.св.')
    print('  (мало — коэффициент один для всех k; много — зависит от k)')
