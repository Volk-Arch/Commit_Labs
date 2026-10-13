#!/usr/bin/env python3
"""Тавтология или форма: холономия при ОДИНАКОВОМ местном счёте.

ГЛАВНАЯ ПРЕТЕНЗИЯ к отклику. Нормированная холономия отвечает на
сгущение (-0.40, до 28.7 сигмы) и на разрежение (+0.23, 17.3 сигмы). Но
может быть, она просто МЕРИТ МЕСТНУЮ ПЛОТНОСТЬ, и тогда вся цепочка
«масса -> холономия -> кривизна» есть «плотность -> плотность ->
плотность», то есть тождество.

РЕШАЮЩИЙ ТЕСТ. Нормировка делит на средний масштаб (общее прошлое по
звеньям цикла) — это и есть местный счёт. Значит надо сравнить дырку и
контроль при ОДИНАКОВОМ значении этого знаменателя: разбить события по
нему на корзины и сравнить отношение внутри корзин.

    избыток держится в корзинах  -> величина несёт ФОРМУ
    избыток исчезает             -> это была плотность, тождество

На каждое событие: числитель и знаменатель усредняются по ЕГО циклам
(их около шести, знаменатели не малы), затем делятся — то есть
отношение средних на уровне события.
"""
import math
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from topo import cone_rel
from holo2 import links_of, cycles_through, holonomy, antisym_causal
from response import warped_diamond

N, T, RAD, SAMP, NCYC, SEEDS = 2600, 2.0, 0.24, 350, 6, 20
LO, HI = RAD, RAD + 0.16

def events(X, rs):
    """На каждое событие слоя: (числитель, знаменатель)."""
    R = cone_rel(X); L = links_of(R); U = L | L.T
    A = R.astype(np.float32)
    CP = A @ A.T
    G = antisym_causal((CP - (A.T @ A)).astype(np.float64), R)
    rr = np.linalg.norm(X[:, 1:], axis=1)
    pool = np.flatnonzero((rr >= LO) & (rr < HI) & (U.sum(1) >= 3))
    out = []
    if len(pool) >= 25:
        pool = rs.choice(pool, min(SAMP, len(pool)), replace=False)
        for v in pool:
            v = int(v)
            cyc = cycles_through(U, v, NCYC, rs)
            if len(cyc) < 2:
                continue
            num = np.mean([abs(holonomy(c, G)) for c in cyc])
            den = np.mean([np.mean([CP[c[k], c[(k + 1) % len(c)]]
                                   for k in range(len(c))]) for c in cyc])
            out.append((num, max(float(den), 1.0)))
    del R, L, U
    return np.array(out) if out else None

H, C = [], []
for t in range(SEEDS):
    rs = np.random.default_rng(7001000 + 977 * t)
    X0 = warped_diamond(N, T, 0.0, 0.45, rs)
    rr = np.linalg.norm(X0[:, 1:], axis=1)
    ins = np.flatnonzero(rr < RAD)
    keep = np.ones(len(X0), bool); keep[ins] = False
    a = events(X0[keep], np.random.default_rng(67 + t))
    ii = rs.choice(len(X0), len(X0) - len(ins), replace=False)
    b = events(X0[np.sort(ii)], np.random.default_rng(67 + t))
    del X0
    if a is not None:
        H.append(a)
    if b is not None:
        C.append(b)
H = np.vstack(H); C = np.vstack(C)
print(f'событий: дырка {len(H)}, контроль {len(C)}')
print(f'знаменатель (местный счёт): дырка {H[:,1].mean():.1f}, '
      f'контроль {C[:,1].mean():.1f}')
print(f'отношение по средним: дырка {H[:,0].sum()/H[:,1].sum():.4f}, '
      f'контроль {C[:,0].sum()/C[:,1].sum():.4f}, '
      f'разница {H[:,0].sum()/H[:,1].sum() - C[:,0].sum()/C[:,1].sum():+.4f}')
print()
print('ВНУТРИ КОРЗИН ПО ЗНАМЕНАТЕЛЮ (одинаковый местный счёт)')
ed = np.percentile(np.concatenate([H[:, 1], C[:, 1]]),
                   [0, 20, 40, 60, 80, 100])
print(f"{'знаменатель':>18}{'дырка n':>9}{'контр n':>9}"
      f"{'отнош. дырка':>14}{'контроль':>11}{'разница':>10}{'сигм':>7}")
num_w, den_w = 0.0, 0.0
for i in range(len(ed) - 1):
    mh = (H[:, 1] >= ed[i]) & (H[:, 1] < ed[i + 1] + (i == len(ed) - 2))
    mc = (C[:, 1] >= ed[i]) & (C[:, 1] < ed[i + 1] + (i == len(ed) - 2))
    if mh.sum() < 20 or mc.sum() < 20:
        continue
    ra = H[mh, 0] / H[mh, 1]
    rb = C[mc, 0] / C[mc, 1]
    d = ra.mean() - rb.mean()
    se = math.hypot(ra.std() / math.sqrt(len(ra)),
                    rb.std() / math.sqrt(len(rb)))
    print(f'{ed[i]:>8.0f}..{ed[i+1]:<9.0f}{mh.sum():>9}{mc.sum():>9}'
          f'{ra.mean():>14.4f}{rb.mean():>11.4f}{d:>+10.4f}'
          f'{abs(d)/max(se,1e-12):>7.1f}')
    num_w += d * mh.sum(); den_w += mh.sum()
if den_w > 0:
    print(f'\n  взвешенная разница ВНУТРИ корзин: {num_w/den_w:+.4f}')
print('  если держится — величина несёт ФОРМУ, не плотность')
