"""Существует ли слабое поле?

Эффект включается с потолка h_add >= 3, что при k=4 требует массы
M ~ 100 событий. Слабое поле требует M << V_bg. На маленьких
интервалах (V_bg ~ 42) эти два условия несовместимы.

Берём большие интервалы (V_bg в сотнях) и смотрим, сохраняется ли
превосходство сосредоточенной массы над рассыпанной при M/V_bg << 1.

Если да — возмущательная гравитация есть, и показатель можно мерить в
правильном режиме. Если нет — эффект существует только когда масса
подавляет интервал, и никакого слабого поля в конструкции нет.
"""
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from chaingrav import build, measure, sample_pairs

N, M, K, R_BALL = 2000, 100, 4, 0.12


def go(min_vol, seed=5):
    Xb, Xbody, Xunif = build(N, M, K, R_BALL, seed)
    rs = np.random.default_rng(seed + 1)
    pairs = sample_pairs(Xb, N, 400, min_vol, rs)
    if len(pairs) < 30:
        return None
    base = measure(Xb, N, pairs)
    bod = measure(Xbody, N, pairs)
    uni = measure(Xunif, N, pairs)
    return base, bod, uni, pairs


print(f'N={N}, масса M={M}, k={K}')
print()
for min_vol in (6, 60, 150, 300):
    r = go(min_vol)
    if r is None:
        print(f'  V_bg >= {min_vol}: мало пар')
        continue
    base, bod, uni, pairs = r
    vbg = base[:, 1].astype(float)
    print(f'  интервалы V_bg >= {min_vol}:  пар {len(pairs)}, '
          f'средний V_bg {vbg.mean():.0f}, M/V_bg {M/vbg.mean():.3f}')
    print(f"{'потолок':>10}{'тело':>12}{'пар':>6}{'контроль':>12}{'пар':>6}"
          f"{'разн., σ':>11}")
    for lo, hi in ((1, 2), (3, 4), (5, 6), (7, 99)):
        cells = []
        for A in (bod, uni):
            msk = (A[:, 3] >= lo) & (A[:, 3] <= hi)
            v = (A[msk, 0] - base[msk, 0]).astype(float)
            cells.append((v.mean() if len(v) else np.nan,
                          v.std() / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                          len(v)))
        (m1, s1, n1), (m2, s2, n2) = cells
        sig = np.hypot(s1, s2)
        z = (m1 - m2) / sig if sig > 0 else np.nan
        lbl = f'{lo}-{hi}' if hi < 99 else f'{lo}+'
        print(f'{lbl:>10}{m1:>12.3f}{n1:>6}{m2:>12.3f}{n2:>6}{z:>11.2f}')
    print()
